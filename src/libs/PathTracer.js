import * as io from "./io.js";

const MB = 1048576;

export class PathTracer {
  constructor(canvas) {
    this.canvas = canvas;
    this.adapter = null;
    this.device = null;
    this.canvasContext = null;

    // 🟢 Default camera parameters
    this.cameraPosition = [0.0, 0.0, 2.5];
    this.cameraQuaternion = [0.0, 0.0, 0.0, 1.0]; // xyzw
  }

  async initialize() {
    await this.getDeviceAndContext();
    await this.loadShaders();
    this.initializeBuffersAndTextures();
    await this.createBindGroupsAndPipelines();
  }

  async buildBVH(tris) {
    const device = this.device;
    const encoder = device.createCommandEncoder();
    const numTriangles = tris.length / 9;

    // ─────────────────────────────
    // 1. Upload triangles buffer
    // ─────────────────────────────
    device.queue.writeBuffer(this.buffers.triangles, 0, tris.buffer);

    // ─────────────────────────────
    // 2. Set builder parameters (UBO)
    // ─────────────────────────────
    const maxDepth  = 10;   // adjustable depth
    const batchSize = 1;    // nodes per thread
    const builderUBO = new Uint32Array([numTriangles, maxDepth, batchSize, 0]);
    device.queue.writeBuffer(this.buffers.builderUBO, 0, builderUBO.buffer);

    // ─────────────────────────────
    // 3. GPU builds BVH entirely (including root)
    // ─────────────────────────────
    let currentLevelStart = 0;
    let currentLevelEnd   = 1;

    for (let level = 0; level < maxDepth; level++) {
      const nextLevelStart = currentLevelEnd;
      const nextLevelEnd   = nextLevelStart + (currentLevelEnd - currentLevelStart) * 2;
      const numNodesThisLevel = currentLevelEnd - currentLevelStart;

      const numWorkgroups = Math.ceil(numNodesThisLevel / batchSize);

      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.BVHBuilder);
      pass.setBindGroup(0, this.bindGroups.BVHBuilder);
      pass.dispatchWorkgroups(numWorkgroups);
      pass.end();

      currentLevelStart = nextLevelStart;
      currentLevelEnd   = nextLevelEnd;
    }

    // ─────────────────────────────
    // 4. Submit all compute passes
    // ─────────────────────────────
    device.queue.submit([encoder.finish()]);

    // Optional: wait for GPU to finish (useful for debugging BVH readback)
    await device.queue.onSubmittedWorkDone();
  }


  async getDeviceAndContext() {
    this.adapter = await navigator.gpu.requestAdapter();
    this.device = await this.adapter.requestDevice();
    this.canvasContext = this.canvas.getContext("webgpu");

    this.canvasContext.configure({
      device: this.device,
      format: "rgba8unorm",
      alphaMode: "premultiplied",
    });
  }

  async loadShaders() {
    const BVHBuilderCode = await io.loadText("./src/shaders/BVHBuilder.wgsl");
    const rendererCode = await io.loadText("./src/shaders/renderer.wgsl");
    const tonemapperCode = await io.loadText("./src/shaders/tonemapper.wgsl");

    this.shaders = {
      BVHBuilder: {
        shader: this.device.createShaderModule({ code: BVHBuilderCode }),
      },
      renderer: {
        shader: this.device.createShaderModule({ code: rendererCode }),
      },
      tonemapper: {
        shader: this.device.createShaderModule({ code: tonemapperCode }),
      },
    };
  }

  initializeBuffersAndTextures() {
    const width = this.canvas.width;
    const height = this.canvas.height;

    this.buffers = {
      readBuffer: this.device.createBuffer({
        size: 16,
        usage:
          GPUBufferUsage.MAP_READ |
          GPUBufferUsage.COPY_DST,
      }),
      rendererUBO: this.device.createBuffer({
        size: 256,
        usage:
          GPUBufferUsage.UNIFORM |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      }),
      builderUBO: this.device.createBuffer({
        size: 256,
        usage:
          GPUBufferUsage.UNIFORM |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      }),
      triangles: this.device.createBuffer({
        size: 4 * MB,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      }),
      BVH: this.device.createBuffer({
        size: 16 * MB,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      }),
    };

    this.textures = {
      outputTexture: this.device.createTexture({
        size: [width, height],
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.COPY_SRC |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST,
      }),
    };
  }

  async createBindGroupsAndPipelines() {
    const device = this.device;
    await device.queue.onSubmittedWorkDone();

    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    // --- Layouts ---
    const bvhBuilderBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    });

    const rendererBindGroupLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            access: "write-only",
            format: "rgba8unorm",
            viewDimension: "2d",
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" },
        },
      ],
    });

    const tonemapperBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      ],
    });

    // --- Pipelines ---
    this.pipelines = {
      BVHBuilder: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bvhBuilderBindGroupLayout] }),
        compute: { module: this.shaders.BVHBuilder.shader, entryPoint: "main" },
      }),

      renderer: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [rendererBindGroupLayout] }),
        compute: { module: this.shaders.renderer.shader, entryPoint: "main" },
      }),

      tonemapper: device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [tonemapperBindGroupLayout] }),
        vertex: { module: this.shaders.tonemapper.shader, entryPoint: "vmain" },
        fragment: {
          module: this.shaders.tonemapper.shader,
          entryPoint: "fmain",
          targets: [{ format: "rgba8unorm" }],
        },
        primitive: { topology: "triangle-list" },
      }),
    };

    // --- Bind Groups ---
    this.bindGroups = {
      BVHBuilder: device.createBindGroup({
        layout: bvhBuilderBindGroupLayout,
        entries: [{ binding: 0, resource: { buffer: this.buffers.BVH } },
                  { binding: 1, resource: { buffer: this.buffers.triangles } },
                  { binding: 2, resource: { buffer: this.buffers.builderUBO } }
        ],
      }),

      renderer: device.createBindGroup({
        layout: rendererBindGroupLayout,
        entries: [
          { binding: 0, resource: this.textures.outputTexture.createView() },
          { binding: 1, resource: { buffer: this.buffers.rendererUBO } },
          { binding: 2, resource: { buffer: this.buffers.triangles } },
          { binding: 3, resource: { buffer: this.buffers.BVH } }
        ],
      }),

      tonemapper: device.createBindGroup({
        layout: tonemapperBindGroupLayout,
        entries: [
          { binding: 0, resource: this.textures.outputTexture.createView() },
          { binding: 1, resource: this.sampler },
        ],
      }),
    };
  }

  async readBVH(sizeInBytes = 1024) {
    const device = this.device;
    const encoder = device.createCommandEncoder();

    // Create a temporary buffer for mapping
    const readBuffer = device.createBuffer({
      size: sizeInBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Copy data from BVH buffer to readBuffer
    encoder.copyBufferToBuffer(this.buffers.BVH, 0, readBuffer, 0, sizeInBytes);
    device.queue.submit([encoder.finish()]);

    // Wait for GPU to complete
    await readBuffer.mapAsync(GPUMapMode.READ);

    // Read data as Float32 or Uint32 depending on structure
    const array = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    console.log("📦 BVH Data:", array);
    return array;
  }

  // 🟢 Called every frame
  async render() {
    const encoder = this.device.createCommandEncoder();

    const trianglesData = new Float32Array([
      // Triangle 1
      -1.0, -1.0, 0.0,
       1.0, -1.0, 0.0,
       0.0,  1.0, 0.0,
      // Triangle 2
      -0.5, -0.5, 1.0,
       0.5, -0.5, 1.0,
       0.0,  0.5, 1.0,
      // Triangle 3
      -1.0,  1.0, -1.0,
       1.0,  1.0, -1.0,
       0.0, -1.0, -1.0,
    ]);

    const numTriangles = trianglesData.length / 9; // 3 vertices per triangle, 3 components per vertex

    // --- UBO (matches WGSL struct) ---
    const UBO = new Float32Array([
      this.canvas.width, this.canvas.height, 0.0, 0.0, // resolution
      ...this.cameraPosition, numTriangles,            // camPos.xyz + numTriangles
      ...this.cameraQuaternion                         // camQuat.xyzw
    ]);

    this.device.queue.writeBuffer(this.buffers.rendererUBO, 0, UBO.buffer);
    this.device.queue.writeBuffer(this.buffers.triangles, 0, trianglesData.buffer);

    await this.buildBVH(new Float32Array(trianglesData));

    function parseAABBs(f32) {
      const aabbs = [];
      for (let i = 0; i < f32.length; i += 6) {
        aabbs.push({
          min: f32.slice(i, i + 3),
          max: f32.slice(i + 3, i + 6),
        });
      }
      return aabbs;
    }

    console.table(parseAABBs(await this.readBVH(256)));


    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.renderer);
      pass.setBindGroup(0, this.bindGroups.renderer);
      pass.dispatchWorkgroups(
        Math.ceil(this.canvas.width / 16),
        Math.ceil(this.canvas.height / 16)
      );
      pass.end();
    }

    // --- Tonemapper ---
    {
      const view = this.canvasContext.getCurrentTexture().createView();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view,
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      });
      pass.setPipeline(this.pipelines.tonemapper);
      pass.setBindGroup(0, this.bindGroups.tonemapper);
      pass.draw(3);
      pass.end();
    }

    this.device.queue.submit([encoder.finish()]);
  }

  setCameraPosition(x, y, z) {
    this.cameraPosition = [x, y, z];
  }

  setCameraQuaternion(x, y, z, w) {
    this.cameraQuaternion = [x, y, z, w];
  }
}
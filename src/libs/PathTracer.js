import * as io from "./io.js";

const MB = 1048576;

export class PathTracer {
  constructor(canvas) {
    this.canvas = canvas;
    this.adapter = null;
    this.device = null;
    this.canvasContext = null;

    // Camera
    this.cameraPosition = [0.0, 0.0, 2.5];
    this.cameraQuaternion = [0.0, 0.0, 0.0, 1.0]; // xyzw

    // BVH build controls
    this.maxDepth = 4; // adjust as you like
    this.batchSize = 1;

    // Derived BVH sizing (filled in initializeBuffersAndTextures)
    this.numNodes = 0;
    this.bvhFloats = 0;
    this.bvhSizeBytes = 0;
  }

  // ---------- utility: compute sizing from depth ----------
  computeBVHSizing(maxDepth) {
    const numNodes = (1 << (maxDepth + 1)) - 1; // full binary tree
    const bvhFloats = numNodes * 6 + 1; // 6 floats/node + 1 float metadata (numNodes)
    const bvhSizeBytes = bvhFloats * 4; // bytes
    return { numNodes, bvhFloats, bvhSizeBytes };
  }

  async initialize() {
    await this.getDeviceAndContext();
    await this.loadShaders();
    this.initializeBuffersAndTextures();
    await this.createBindGroupsAndPipelines();
  }

  // ---------- BVH builder ----------
  async buildBVH(tris) {
    const device = this.device;
    const encoder = device.createCommandEncoder();
    const numTriangles = tris.length / 9;

    // 0) Ensure BVH buffer sized for current depth
    this.ensureBVHBuffer(this.maxDepth);

    // 1) Upload triangles
    device.queue.writeBuffer(this.buffers.triangles, 0, tris.buffer);

    // 2) Builder UBO (numTriangles, maxDepth, batchSize, padding)
    const builderUBO = new Uint32Array([
      numTriangles,
      this.maxDepth,
      this.batchSize,
      0,
    ]);
    device.queue.writeBuffer(this.buffers.builderUBO, 0, builderUBO.buffer);

    // 3) Dispatch the builder once (shader does breadth-first internally)
    {
      const groups = Math.ceil((1 << this.maxDepth) / this.batchSize);
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.BVHBuilder);
      pass.setBindGroup(0, this.bindGroups.BVHBuilder);
      pass.dispatchWorkgroups(groups);
      pass.end();
    }

    // 4) Submit
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  // Allocate/reallocate BVH buffer exactly to required size
  ensureBVHBuffer(maxDepth) {
    const sizing = this.computeBVHSizing(maxDepth);
    const needRealloc =
      !this.buffers?.BVH || this.bvhSizeBytes !== sizing.bvhSizeBytes;

    this.numNodes = sizing.numNodes;
    this.bvhFloats = sizing.bvhFloats;
    this.bvhSizeBytes = sizing.bvhSizeBytes;

    if (!needRealloc) return;

    // (Re)create BVH buffer
    if (this.buffers?.BVH && this.buffers.BVH.destroy) {
      this.buffers.BVH.destroy();
    }

    this.buffers.BVH = this.device.createBuffer({
      size: this.bvhSizeBytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    // Rebuild any bind groups that reference BVH
    this.updateBindGroups();
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

    // Precompute BVH size for the initial depth
    const sizing = this.computeBVHSizing(this.maxDepth);
    this.numNodes = sizing.numNodes;
    this.bvhFloats = sizing.bvhFloats;
    this.bvhSizeBytes = sizing.bvhSizeBytes;

    this.buffers = {
      readBuffer: this.device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
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
        size: this.bvhSizeBytes, // exact size
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
    this.layouts = {
      bvhBuilder: device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" },
          },
        ],
      }),
      renderer: device.createBindGroupLayout({
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
      }),
      tonemapper: device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            texture: { sampleType: "float" },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            sampler: { type: "filtering" },
          },
        ],
      }),
    };

    // --- Pipelines ---
    this.pipelines = {
      BVHBuilder: device.createComputePipeline({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [this.layouts.bvhBuilder],
        }),
        compute: { module: this.shaders.BVHBuilder.shader, entryPoint: "main" },
      }),
      renderer: device.createComputePipeline({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [this.layouts.renderer],
        }),
        compute: { module: this.shaders.renderer.shader, entryPoint: "main" },
      }),
      tonemapper: device.createRenderPipeline({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [this.layouts.tonemapper],
        }),
        vertex: { module: this.shaders.tonemapper.shader, entryPoint: "vmain" },
        fragment: {
          module: this.shaders.tonemapper.shader,
          entryPoint: "fmain",
          targets: [{ format: "rgba8unorm" }],
        },
        primitive: { topology: "triangle-list" },
      }),
    };

    this.updateBindGroups();
  }

  // Recreate bind groups that reference BVH when BVH is resized
  updateBindGroups() {
    const device = this.device;

    this.bindGroups = {
      BVHBuilder: device.createBindGroup({
        layout: this.layouts.bvhBuilder,
        entries: [
          { binding: 0, resource: { buffer: this.buffers.BVH } },
          { binding: 1, resource: { buffer: this.buffers.triangles } },
          { binding: 2, resource: { buffer: this.buffers.builderUBO } },
        ],
      }),
      renderer: device.createBindGroup({
        layout: this.layouts.renderer,
        entries: [
          { binding: 0, resource: this.textures.outputTexture.createView() },
          { binding: 1, resource: { buffer: this.buffers.rendererUBO } },
          { binding: 2, resource: { buffer: this.buffers.triangles } },
          { binding: 3, resource: { buffer: this.buffers.BVH } },
        ],
      }),
      tonemapper: device.createBindGroup({
        layout: this.layouts.tonemapper,
        entries: [
          { binding: 0, resource: this.textures.outputTexture.createView() },
          { binding: 1, resource: this.sampler },
        ],
      }),
    };
  }

  // Map the whole BVH (or a custom slice) for debugging
  async readBVH(sizeInBytes = this.bvhSizeBytes) {
    const device = this.device;
    const encoder = device.createCommandEncoder();

    const readBuffer = device.createBuffer({
      size: sizeInBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    encoder.copyBufferToBuffer(this.buffers.BVH, 0, readBuffer, 0, sizeInBytes);
    device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const array = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();

    console.log("📦 BVH Data:", array);
    return array;
  }

  // ---------- per-frame ----------
  async render() {
    // Example scene: 3 triangles
    const trianglesData = new Float32Array([
      // Front (z = +1)
      -1, -1,  1,
      1, -1,  1,
      1,  1,  1,
      -1, -1,  1,
      1,  1,  1,
      -1,  1,  1,

      // Back (z = -1)
      1, -1, -1,
      -1, -1, -1,
      -1,  1, -1,
      1, -1, -1,
      -1,  1, -1,
      1,  1, -1,

      // Left (x = -1)
      -1, -1, -1,
      -1, -1,  1,
      -1,  1,  1,
      -1, -1, -1,
      -1,  1,  1,
      -1,  1, -1,

      // Right (x = +1)
      1, -1,  1,
      1, -1, -1,
      1,  1, -1,
      1, -1,  1,
      1,  1, -1,
      1,  1,  1,

      // Top (y = +1)
      -1,  1,  1,
      1,  1,  1,
      1,  1, -1,
      -1,  1,  1,
      1,  1, -1,
      -1,  1, -1,

      // Bottom (y = -1)
      -1, -1, -1,
      1, -1, -1,
      1, -1,  1,
      -1, -1, -1,
      1, -1,  1,
      -1, -1,  1,
    ]);

    const numTriangles = trianglesData.length / 9;

    // Build BVH first (allocates the exact BVH size if needed)
    await this.buildBVH(new Float32Array(trianglesData));

    // (Optional) read back to inspect
    const bvhData = await this.readBVH();
    const numNodesFromSize = Math.floor((bvhData.length - 1) / 6); // minus metadata float
    console.log(`${numNodesFromSize} BVH nodes (derived from buffer length)`);

    // Write renderer UBO AFTER build so numNodes is in sync
    const UBO = new Float32Array([
      this.canvas.width,
      this.canvas.height,
      0.0,
      0.0, // resolution
      ...this.cameraPosition,
      numTriangles, // camPos.xyz + numTriangles
      ...this.cameraQuaternion, // camQuat.xyzw
      this.numNodes,
      0.0,
      0.0,
      0.0, // nodes.x = numNodes
    ]);
    this.device.queue.writeBuffer(this.buffers.rendererUBO, 0, UBO.buffer);
    this.device.queue.writeBuffer(
      this.buffers.triangles,
      0,
      trianglesData.buffer
    );

    // ---------- render compute ----------
    const encoder = this.device.createCommandEncoder();

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

    // ---------- tonemapper ----------
    {
      const view = this.canvasContext.getCurrentTexture().createView();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          },
        ],
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
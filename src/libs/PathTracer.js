import * as io from "./io.js";

const MB = 1048576;
const BVH_WORKGROUP_SIZE = 256;

export class PathTracer {
  constructor(canvas) {
    this.canvas = canvas;
    this.adapter = null;
    this.device = null;
    this.canvasContext = null;

    // Camera
    this.cameraPosition = [0.0, 0.0, 3.5];
    this.cameraQuaternion = [0.0, 0.0, 0.0, 1.0];

    // 8-ary BVH config
    // Depth 6 → capacity 8^6 = 262,144 triangles (moderate scenes)
    this.maxDepth = 7;

    this.numNodes = 0;
    this.bvhFloats = 0;
    this.bvhSizeBytes = 0;
    this.currentDepth = 0;   // actual depth used for current scene

    this.buffers = {};
    this.textures = {};
    this.layouts = {};
    this.pipelines = {};
    this.bindGroups = {};

    // Default mesh (simple cube as 4 triangles, already normalized)
    this.trianglesData = new Float32Array([
      1, 1, 1, -1, -1, 1, -1, 1, -1,
      1, 1, 1, -1, 1, -1, 1, -1, -1,
      1, 1, 1, 1, -1, -1, -1, -1, 1,
      -1, -1, 1, 1, -1, -1, -1, 1, -1,
    ]);
  }

  // Sizing for perfect 8-ary tree: totalNodes = (8^(d+1) − 1) / 7
  computeBVHSizing(maxDepth) {
    const pow8Next = Math.pow(8, maxDepth + 1);
    const numNodes = Math.floor((pow8Next - 1) / 7);

    const bvhFloats = 1 + numNodes * 8; // 1 metadata + 8 per node
    const bvhSizeBytes = bvhFloats * 4;

    return { numNodes, bvhFloats, bvhSizeBytes };
  }

  async initialize() {
    await this.getDeviceAndContext();
    await this.loadShaders();
    this.initializeBuffersAndTextures();
    await this.createBindGroupsAndPipelines();
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
    const rendererCode   = await io.loadText("./src/shaders/renderer.wgsl");
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
    const width  = this.canvas.width;
    const height = this.canvas.height;

    this.buffers = {
      rendererUBO: this.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      builderUBO: this.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      triangles: this.device.createBuffer({
        // 32 MB triangle pool – JS side enforces not to overflow this
        size: 32 * MB,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      }),
      BVH: null, // created by ensureBVHBuffer()
    };

    // Minimal BVH buffer so that bind groups can be created;
    // real size is decided in buildBVH() depending on scene.
    this.ensureBVHBuffer(1);

    this.textures = {
      outputTexture: this.device.createTexture({
        size: [width, height],
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING,
      }),
    };
  }

  ensureBVHBuffer(depth) {
    const sizing = this.computeBVHSizing(depth);

    const currentSize = this.buffers.BVH ? this.buffers.BVH.size : 0;

    // Reuse if big enough; only update metadata
    if (this.buffers.BVH && currentSize >= sizing.bvhSizeBytes) {
      this.numNodes     = sizing.numNodes;
      this.bvhFloats    = sizing.bvhFloats;
      this.bvhSizeBytes = sizing.bvhSizeBytes;
      return;
    }

    console.log(
      `Reallocating BVH Buffer for depth ${depth}: ${(sizing.bvhSizeBytes /
        1024 /
        1024).toFixed(2)} MB`
    );

    if (this.buffers.BVH) this.buffers.BVH.destroy();

    this.numNodes     = sizing.numNodes;
    this.bvhFloats    = sizing.bvhFloats;
    this.bvhSizeBytes = sizing.bvhSizeBytes;

    this.buffers.BVH = this.device.createBuffer({
      size: this.bvhSizeBytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    // If layouts/pipelines are not created yet this is a no-op.
    this.updateBindGroups();
  }

  async createBindGroupsAndPipelines() {
    const device = this.device;
    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

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

    this.pipelines = {
      BVHBuilder: device.createComputePipeline({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [this.layouts.bvhBuilder],
        }),
        compute: {
          module: this.shaders.BVHBuilder.shader,
          entryPoint: "main",
        },
      }),
      renderer: device.createComputePipeline({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [this.layouts.renderer],
        }),
        compute: {
          module: this.shaders.renderer.shader,
          entryPoint: "main",
        },
      }),
      tonemapper: device.createRenderPipeline({
        layout: device.createPipelineLayout({
          bindGroupLayouts: [this.layouts.tonemapper],
        }),
        vertex: {
          module: this.shaders.tonemapper.shader,
          entryPoint: "vmain",
        },
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

  updateBindGroups() {
    // During very early init, layouts might not exist yet
    if (!this.buffers.BVH || !this.layouts || !this.layouts.bvhBuilder) return;

    this.bindGroups.BVHBuilder = this.device.createBindGroup({
      layout: this.layouts.bvhBuilder,
      entries: [
        { binding: 0, resource: { buffer: this.buffers.BVH } },
        { binding: 1, resource: { buffer: this.buffers.triangles } },
        { binding: 2, resource: { buffer: this.buffers.builderUBO } },
      ],
    });

    this.bindGroups.renderer = this.device.createBindGroup({
      layout: this.layouts.renderer,
      entries: [
        {
          binding: 0,
          resource: this.textures.outputTexture.createView(),
        },
        { binding: 1, resource: { buffer: this.buffers.rendererUBO } },
        { binding: 2, resource: { buffer: this.buffers.triangles } },
        { binding: 3, resource: { buffer: this.buffers.BVH } },
      ],
    });

    this.bindGroups.tonemapper = this.device.createBindGroup({
      layout: this.layouts.tonemapper,
      entries: [
        {
          binding: 0,
          resource: this.textures.outputTexture.createView(),
        },
        { binding: 1, resource: this.sampler },
      ],
    });
  }

  async buildBVH(trianglesData) {
    const start = performance.now();
    const device = this.device;
    if (!device) {
      console.error("Device not initialized");
      return;
    }

    const numTriangles = trianglesData.length / 9;
    const maxDepth = this.maxDepth;

    // smallest depth such that 8^depth >= numTriangles (capped by maxDepth)
    let depthUsed = 0;
    let capacity = 1;
    while (capacity < numTriangles && depthUsed < maxDepth) {
      depthUsed += 1;
      capacity *= 8;
    }

    this.currentDepth = depthUsed;

    console.log(
      `Building BVH8: ${numTriangles} tris, Depth ${depthUsed} (Capacity: ${capacity})`
    );

    // allocate BVH buffer for exactly this depth
    this.ensureBVHBuffer(depthUsed);

    // upload triangles
    if (!this.buffers.triangles) {
      console.error("Triangle buffer missing");
      return;
    }
    device.queue.writeBuffer(
      this.buffers.triangles,
      0,
      new Float32Array(trianglesData)
    );

    // BVH[0] = total node count (float). Renderer reads this.
    device.queue.writeBuffer(
      this.buffers.BVH,
      0,
      new Float32Array([this.numNodes])
    );

    if (numTriangles === 0) {
      console.warn("No triangles to build BVH for.");
      await device.queue.onSubmittedWorkDone();
      return;
    }

    const encoder = device.createCommandEncoder();

    const totalNodes = this.numNodes;
    const workgroups = Math.ceil(totalNodes / BVH_WORKGROUP_SIZE);

    // UBO for BVHBuilder: x = numTris, y = maxDepth used
    const uboData = new Uint32Array([
      numTriangles,
      depthUsed,
      0,
      0,
    ]);
    device.queue.writeBuffer(this.buffers.builderUBO, 0, uboData);

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.BVHBuilder);
    pass.setBindGroup(0, this.bindGroups.BVHBuilder);
    pass.dispatchWorkgroups(workgroups);
    pass.end();

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const end = performance.now();
    console.log(
      `BVH Build Complete: depth=${depthUsed}, nodes=${totalNodes}, time=${(
        end - start
      ).toFixed(2)} ms`
    );
  }

  async readBVH() {
    if (!this.buffers.BVH) return new Float32Array(0);
    const size = this.bvhSizeBytes;

    const readBuffer = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const cmd = this.device.createCommandEncoder();
    cmd.copyBufferToBuffer(this.buffers.BVH, 0, readBuffer, 0, size);
    this.device.queue.submit([cmd.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const arr = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    return arr;
  }

  async setScene(scene) {
    // scene.getTrianglesFloat32() should return Float32Array [x,y,z] * 3 per tri
    this.trianglesData = scene.getTrianglesFloat32();
    await this.buildBVH(this.trianglesData);
  }

  async render() {
    if (!this.buffers.BVH) return;

    const numTriangles = this.trianglesData.length / 9;

    const depthUsed = this.currentDepth;
    let fov = 70.0 * Math.PI / 180;
    let focal = 1.0 / Math.tan(0.5 * fov);

    const UBO = new Float32Array([
      // resolution
      this.canvas.width,
      this.canvas.height,
      focal,
      this.canvas.width / this.canvas.height,

      // camPos + numTris
      this.cameraPosition[0],
      this.cameraPosition[1],
      this.cameraPosition[2],
      numTriangles,

      // camera quaternion
      this.cameraQuaternion[0],
      this.cameraQuaternion[1],
      this.cameraQuaternion[2],
      this.cameraQuaternion[3],

      // numNodes + depth (node count also lives in BVH[0])
      this.numNodes,
      depthUsed,
      0,
      0,
    ]);

    this.device.queue.writeBuffer(this.buffers.rendererUBO, 0, UBO);

    const encoder = this.device.createCommandEncoder();

    // Compute (path tracer)
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.renderer);
    pass.setBindGroup(0, this.bindGroups.renderer);
    const wgX = Math.ceil(this.canvas.width / 16);
    const wgY = Math.ceil(this.canvas.height / 16);
    pass.dispatchWorkgroups(wgX, wgY);
    console.log(`Dispatching Renderer: ${wgX * wgY} workgroups`);
    pass.end();

    // Tonemap to swapchain
    const view = this.canvasContext.getCurrentTexture().createView();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    renderPass.setPipeline(this.pipelines.tonemapper);
    renderPass.setBindGroup(0, this.bindGroups.tonemapper);
    renderPass.draw(3);
    renderPass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  setCameraPosition(x, y, z) {
    this.cameraPosition = [x, y, z];
  }

  setCameraQuaternion(x, y, z, w) {
    this.cameraQuaternion = [x, y, z, w];
  }
}
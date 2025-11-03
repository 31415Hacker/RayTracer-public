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
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
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
        entries: [{ binding: 0, resource: { buffer: this.buffers.BVH } }],
      }),

      renderer: device.createBindGroup({
        layout: rendererBindGroupLayout,
        entries: [
          { binding: 0, resource: this.textures.outputTexture.createView() },
          { binding: 1, resource: { buffer: this.buffers.rendererUBO } },
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

  // 🟢 Called every frame
  render() {
    const encoder = this.device.createCommandEncoder();

    // --- UBO (matches WGSL struct) ---
    const UBO = new Float32Array([
      this.canvas.width, this.canvas.height, 0.0, 0.0, // resolution
      ...this.cameraPosition, 0.0,                     // camPos.xyz + pad
      ...this.cameraQuaternion                         // camQuat.xyzw
    ]);

    this.device.queue.writeBuffer(this.buffers.rendererUBO, 0, UBO.buffer);

    // --- Compute passes ---
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.BVHBuilder);
      pass.setBindGroup(0, this.bindGroups.BVHBuilder);
      pass.dispatchWorkgroups(Math.ceil(1024 / 64));
      pass.end();
    }

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
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

    // Config
    // Depth 6 is safe for moderate scenes (up to ~260k triangles capacity in OctTree)
    this.maxDepth = 6; 

    this.numNodes = 0;
    this.bvhFloats = 0;
    this.bvhSizeBytes = 0;
    this.currentDepth = 0;

    this.buffers = {};
    this.textures = {};
    this.layouts = {};
    this.pipelines = {};
    this.bindGroups = {};

    // Default mesh (Cube)
    this.trianglesData = new Float32Array([
      1, 1, 1, -1, -1, 1, -1, 1, -1,
      1, 1, 1, -1, 1, -1, 1, -1, -1,
      1, 1, 1, 1, -1, -1, -1, -1, 1,
      -1, -1, 1, 1, -1, -1, -1, 1, -1,
    ]);
  }

  // Sizing for 8-ary tree: Total Nodes = (8^(d+1) - 1) / 7
  computeBVHSizing(maxDepth) {
    // Use Math.pow for standard sizing
    const pow8Next = Math.pow(8, maxDepth + 1);
    const numNodes = Math.floor((pow8Next - 1) / 7);
    
    const bvhFloats = 1 + numNodes * 6; // 1 metadata + 6 per node
    const bvhSizeBytes = bvhFloats * 4;
    
    return { numNodes, bvhFloats, bvhSizeBytes };
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

    // 1. Calculate exact depth needed
    // We need 8^depth >= numTriangles
    let neededDepth = 0;
    let capacity = 1;
    while (capacity < numTriangles) {
      capacity *= 8;
      neededDepth++;
    }
    
    // Cap at maxDepth, but allow it to be smaller for simple meshes
    //neededDepth = Math.min(neededDepth, this.maxDepth);
    if (neededDepth < 1) neededDepth = 1;

    console.log(`Building BVH8: ${numTriangles} tris, Depth ${neededDepth} (Capacity: ${Math.pow(8, neededDepth)})`);

    // 2. Ensure buffer is big enough for this depth
    this.ensureBVHBuffer(neededDepth);
    this.currentDepth = neededDepth; 

    // 3. Upload Triangles
    // Resize triangle buffer if needed
    if (this.buffers.triangles.size < tris.byteLength) {
        console.log("Resizing Triangle Buffer...");
        this.buffers.triangles.destroy();
        this.buffers.triangles = this.device.createBuffer({
            size: tris.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.updateBindGroups(); // Rebind new buffer
    }
    device.queue.writeBuffer(this.buffers.triangles, 0, tris);

    // 4. Update Builder UBO
    const builderUBO = new Uint32Array([
      numTriangles,
      neededDepth,
      0, 0
    ]);
    device.queue.writeBuffer(this.buffers.builderUBO, 0, builderUBO);

    // 5. Dispatch Builder
    const totalNodes = this.numNodes;
    const numGroups = Math.ceil(totalNodes / BVH_WORKGROUP_SIZE);
    console.log(`Dispatching BVH Builder: ${totalNodes} nodes in ${numGroups} workgroups`);

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.BVHBuilder);
    pass.setBindGroup(0, this.bindGroups.BVHBuilder);
    pass.dispatchWorkgroups(numGroups);
    pass.end();

    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  ensureBVHBuffer(depth) {
    const sizing = this.computeBVHSizing(depth);
    
    // Reallocate if buffer is null or too small
    const currentSize = this.buffers.BVH ? this.buffers.BVH.size : 0;
    
    if (this.buffers.BVH && currentSize >= sizing.bvhSizeBytes) {
        // Reuse existing buffer, just update sizing metadata for renderer
        this.numNodes = sizing.numNodes;
        this.bvhFloats = sizing.bvhFloats;
        this.bvhSizeBytes = sizing.bvhSizeBytes;
        return;
    }

    console.log(`Reallocating BVH Buffer for depth ${depth}: ${(sizing.bvhSizeBytes / 1024 / 1024).toFixed(2)} MB`);

    if (this.buffers.BVH) this.buffers.BVH.destroy();

    this.numNodes = sizing.numNodes;
    this.bvhFloats = sizing.bvhFloats;
    this.bvhSizeBytes = sizing.bvhSizeBytes;

    this.buffers.BVH = this.device.createBuffer({
      size: this.bvhSizeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

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
      BVHBuilder: { shader: this.device.createShaderModule({ code: BVHBuilderCode }) },
      renderer: { shader: this.device.createShaderModule({ code: rendererCode }) },
      tonemapper: { shader: this.device.createShaderModule({ code: tonemapperCode }) },
    };
  }

  initializeBuffersAndTextures() {
    const width = this.canvas.width;
    const height = this.canvas.height;
    
    this.buffers = {
      rendererUBO: this.device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      builderUBO: this.device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      triangles: this.device.createBuffer({ size: 16 * MB, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }),
      BVH: null // Will be created in ensureBVHBuffer
    };

    // This allocates the buffer, but updateBindGroups inside it will now gracefully fail 
    // because layouts aren't made yet. They will be bound later in createBindGroupsAndPipelines.
    this.ensureBVHBuffer(4); 

    this.textures = {
      outputTexture: this.device.createTexture({
        size: [width, height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      }),
    };
  }

  async createBindGroupsAndPipelines() {
    const device = this.device;
    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

    this.layouts = {
      bvhBuilder: device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        ],
      }),
      renderer: device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm", viewDimension: "2d" } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        ],
      }),
      tonemapper: device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        ],
      }),
    };

    this.pipelines = {
      BVHBuilder: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.layouts.bvhBuilder] }),
        compute: { module: this.shaders.BVHBuilder.shader, entryPoint: "main" },
      }),
      renderer: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.layouts.renderer] }),
        compute: { module: this.shaders.renderer.shader, entryPoint: "main" },
      }),
      tonemapper: device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.layouts.tonemapper] }),
        vertex: { module: this.shaders.tonemapper.shader, entryPoint: "vmain" },
        fragment: { module: this.shaders.tonemapper.shader, entryPoint: "fmain", targets: [{ format: "rgba8unorm" }] },
        primitive: { topology: "triangle-list" },
      }),
    };

    // Now that layouts exist, we can actually create the bind groups
    this.updateBindGroups();
  }

  updateBindGroups() {
    // CRITICAL FIX: 
    // During initialization, we create buffers BEFORE layouts.
    // ensureBVHBuffer calls this method. If layouts don't exist yet, we must exit.
    // They will be created later in createBindGroupsAndPipelines().
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
        { binding: 0, resource: this.textures.outputTexture.createView() },
        { binding: 1, resource: { buffer: this.buffers.rendererUBO } },
        { binding: 2, resource: { buffer: this.buffers.triangles } },
        { binding: 3, resource: { buffer: this.buffers.BVH } },
      ],
    });

    this.bindGroups.tonemapper = this.device.createBindGroup({
      layout: this.layouts.tonemapper,
      entries: [
        { binding: 0, resource: this.textures.outputTexture.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });
  }

  async readBVH() {
    if (!this.buffers.BVH) return new Float32Array(0);
    const size = this.bvhSizeBytes;
    
    const readBuffer = this.device.createBuffer({
        size: size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
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
    this.trianglesData = scene.getTrianglesFloat32();
    await this.buildBVH(this.trianglesData);
  }

  async render() {
    if (!this.buffers.BVH) return;

    const numTriangles = this.trianglesData.length / 9;
    
    // Rebuild BVH every frame (can be optimized later)
    // await this.buildBVH(this.trianglesData);

    const depthUsed = this.currentDepth; 

    const UBO = new Float32Array([
      this.canvas.width, this.canvas.height, 0, 0,
      this.cameraPosition[0], this.cameraPosition[1], this.cameraPosition[2], numTriangles,
      this.cameraQuaternion[0], this.cameraQuaternion[1], this.cameraQuaternion[2], this.cameraQuaternion[3],
      this.numNodes, depthUsed, 0, 0
    ]);

    this.device.queue.writeBuffer(this.buffers.rendererUBO, 0, UBO);

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.renderer);
    pass.setBindGroup(0, this.bindGroups.renderer);
    pass.dispatchWorkgroups(Math.ceil(this.canvas.width / 16), Math.ceil(this.canvas.height / 16));
    pass.end();

    // Tonemap
    const view = this.canvasContext.getCurrentTexture().createView();
    const renderPass = encoder.beginRenderPass({
        colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }]
    });
    renderPass.setPipeline(this.pipelines.tonemapper);
    renderPass.setBindGroup(0, this.bindGroups.tonemapper);
    renderPass.draw(3);
    renderPass.end();

    this.device.queue.submit([encoder.finish()]);
  }
  
  setCameraPosition(x, y, z) { this.cameraPosition = [x, y, z]; }
  setCameraQuaternion(x, y, z, w) { this.cameraQuaternion = [x, y, z, w]; }
}
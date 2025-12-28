import * as io from "./io.js";

const MB = 1048576;

const BVH_WORKGROUP_SIZE = 256;

// LBVH2 layout produced by GPU builder
const NODE2_STRIDE_U32 = 6;
const LEAF_FLAG = 0x80000000 >>> 0;
const INVALID = 0xffffffff >>> 0;

// Final BVH4 layout consumed by renderer
const NODE4_STRIDE_U32 = 8;

// ---- FP16 helpers (local to this function) ----
const f16ToF32 = (h) => {
  const s = (h & 0x8000) << 16;
  let e = (h >> 10) & 0x1f;
  let m = h & 0x03ff;

  if (e === 0) {
    if (m === 0) return new Float32Array(new Uint32Array([s]).buffer)[0];
    // subnormal
    e = 1;
    while ((m & 0x0400) === 0) {
      m <<= 1;
      e--;
    }
    m &= 0x03ff;
  } else if (e === 31) {
    return new Float32Array(new Uint32Array([s | 0x7f800000 | (m << 13)]).buffer)[0];
  }

  const u =
    s |
    ((e + 112) << 23) |
    (m << 13);

  return new Float32Array(new Uint32Array([u]).buffer)[0];
};

const f32ToF16 = (v) => {
  const u = new Uint32Array(new Float32Array([v]).buffer)[0];
  const s = (u >> 16) & 0x8000;
  let e = ((u >> 23) & 0xff) - 112;
  let m = (u >> 13) & 0x03ff;

  if (e <= 0) return s;
  if (e >= 31) return s | 0x7c00;
  return s | (e << 10) | m;
};

const pack16x2 = (a, b) =>
  (f32ToF16(a) | (f32ToF16(b) << 16)) >>> 0;

const unpack16x2 = (u, idx) =>
  f16ToF32((u >>> (idx * 16)) & 0xffff);

const BVH_EPSILON = 5e-4;

export class PathTracer {
  constructor(canvas) {
    this.canvas = canvas;
    this.adapter = null;
    this.device = null;
    this.canvasContext = null;

    // Camera
    this.cameraPosition = [0.0, 0.0, 3.5];
    this.cameraQuaternion = [0.0, 0.0, 0.0, 1.0];

    this.buffers = {};
    this.textures = {};
    this.layouts = {};
    this.pipelines = {};
    this.bindGroups = {};

    this.frameCount = 0;

    // Default mesh
    this.trianglesData = new Float32Array([
      1, 1, 1, -1, -1, 1, -1, 1, -1,
      1, 1, 1, -1, 1, -1, 1, -1, -1,
      1, 1, 1, 1, -1, -1, -1, -1, 1,
      -1, -1, 1, 1, -1, -1, -1, 1, -1,
    ]);

    // bookkeeping
    this.bvh2SizeBytes = 0;
    this.bvh4SizeBytes = 0;
    this.auxSizes = {
      morton: 0,
      triIdx: 0,
      parent: 0,
      flags: 0,
    };
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
      BVHBuilder: { shader: this.device.createShaderModule({ code: BVHBuilderCode }) },
      renderer: { shader: this.device.createShaderModule({ code: rendererCode }) },
      tonemapper: { shader: this.device.createShaderModule({ code: tonemapperCode }) },
    };
  }

  initializeBuffersAndTextures() {
    const width = this.canvas.width;
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
        size: 32 * MB,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      }),

      // LBVH2 build outputs (GPU)
      BVH2: null,

      // Final BVH4 used by renderer (GPU)
      BVH: null,

      // LBVH aux (GPU)
      mortonSorted: null,
      triIndexSorted: null,
      parent: null,
      buildFlags: null,
    };

    // Create minimal buffers so bind groups can be created early
    this.ensureLBVHAuxBuffers(1, 1);
    this.ensureBVH2Buffer(1);
    this.ensureBVH4Buffer(1);

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

  ensureLBVHAuxBuffers(numTris, numNodes2) {
    const device = this.device;

    const mortonBytes = Math.max(1, numTris) * 4;
    const triIdxBytes = Math.max(1, numTris) * 4;
    const parentBytes = Math.max(1, numNodes2) * 4;
    const flagsBytes = Math.max(1, Math.max(numTris - 1, 1)) * 4;

    const needRealloc =
      !this.buffers.mortonSorted ||
      this.auxSizes.morton < mortonBytes ||
      this.auxSizes.triIdx < triIdxBytes ||
      this.auxSizes.parent < parentBytes ||
      this.auxSizes.flags < flagsBytes;

    if (!needRealloc) return;

    if (this.buffers.mortonSorted) this.buffers.mortonSorted.destroy();
    if (this.buffers.triIndexSorted) this.buffers.triIndexSorted.destroy();
    if (this.buffers.parent) this.buffers.parent.destroy();
    if (this.buffers.buildFlags) this.buffers.buildFlags.destroy();

    this.buffers.mortonSorted = device.createBuffer({
      size: mortonBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.buffers.triIndexSorted = device.createBuffer({
      size: triIdxBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.buffers.parent = device.createBuffer({
      size: parentBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.buffers.buildFlags = device.createBuffer({
      size: flagsBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.auxSizes = {
      morton: mortonBytes,
      triIdx: triIdxBytes,
      parent: parentBytes,
      flags: flagsBytes,
    };

    this.updateBindGroups();
  }

  computeBVH2Sizing(numTris) {
    if (numTris <= 0) return { numNodes2: 0, bytes: 4 };
    const numNodes2 = 2 * numTris - 1;
    const u32s = 1 + numNodes2 * NODE2_STRIDE_U32;
    return { numNodes2, bytes: u32s * 4 };
  }

  computeBVH4Sizing(numNodes4) {
    if (numNodes4 <= 0) return { bytes: 4 };
    const u32s = 1 + numNodes4 * NODE4_STRIDE_U32;
    return { bytes: u32s * 4 };
  }

  ensureBVH2Buffer(numTris) {
    const device = this.device;
    const { numNodes2, bytes } = this.computeBVH2Sizing(numTris);

    if (this.buffers.BVH2 && this.bvh2SizeBytes >= bytes) return;

    if (this.buffers.BVH2) this.buffers.BVH2.destroy();

    this.bvh2SizeBytes = Math.max(4, bytes);

    this.buffers.BVH2 = device.createBuffer({
      size: this.bvh2SizeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Write BVH2[0] = numNodes2
    device.queue.writeBuffer(this.buffers.BVH2, 0, new Uint32Array([numNodes2 >>> 0]));

    this.updateBindGroups();
  }

  ensureBVH4Buffer(numNodes4) {
    const device = this.device;
    const { bytes } = this.computeBVH4Sizing(numNodes4);

    if (this.buffers.BVH && this.bvh4SizeBytes >= bytes) return;

    if (this.buffers.BVH) this.buffers.BVH.destroy();

    this.bvh4SizeBytes = Math.max(4, bytes);

    this.buffers.BVH = device.createBuffer({
      size: this.bvh4SizeBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // BVH[0] set later once we know numNodes4
    this.device.queue.writeBuffer(this.buffers.BVH, 0, new Uint32Array([0]));

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
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // BVH2
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // triangles
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // mortonSorted
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // triIndexSorted
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // parent
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },            // buildFlags
          { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },            // ubo
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
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // BVH4
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
      BVHBuildInternal: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.layouts.bvhBuilder] }),
        compute: { module: this.shaders.BVHBuilder.shader, entryPoint: "buildInternal" },
      }),
      BVHBuildLeaves: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.layouts.bvhBuilder] }),
        compute: { module: this.shaders.BVHBuilder.shader, entryPoint: "buildLeaves" },
      }),

      renderer: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.layouts.renderer] }),
        compute: { module: this.shaders.renderer.shader, entryPoint: "main" },
      }),

      tonemapper: device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.layouts.tonemapper] }),
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

  updateBindGroups() {
    // early init safety
    if (!this.layouts || !this.device) return;

    if (
      this.layouts.bvhBuilder &&
      this.buffers.BVH2 &&
      this.buffers.triangles &&
      this.buffers.mortonSorted &&
      this.buffers.triIndexSorted &&
      this.buffers.parent &&
      this.buffers.buildFlags &&
      this.buffers.builderUBO
    ) {
      this.bindGroups.BVHBuilder = this.device.createBindGroup({
        layout: this.layouts.bvhBuilder,
        entries: [
          { binding: 0, resource: { buffer: this.buffers.BVH2 } },
          { binding: 1, resource: { buffer: this.buffers.triangles } },
          { binding: 2, resource: { buffer: this.buffers.mortonSorted } },
          { binding: 3, resource: { buffer: this.buffers.triIndexSorted } },
          { binding: 4, resource: { buffer: this.buffers.parent } },
          { binding: 5, resource: { buffer: this.buffers.buildFlags } },
          { binding: 6, resource: { buffer: this.buffers.builderUBO } },
        ],
      });
    }

    if (this.layouts.renderer && this.buffers.rendererUBO && this.buffers.triangles && this.buffers.BVH && this.textures.outputTexture) {
      this.bindGroups.renderer = this.device.createBindGroup({
        layout: this.layouts.renderer,
        entries: [
          { binding: 0, resource: this.textures.outputTexture.createView() },
          { binding: 1, resource: { buffer: this.buffers.rendererUBO } },
          { binding: 2, resource: { buffer: this.buffers.triangles } },
          { binding: 3, resource: { buffer: this.buffers.BVH } },
        ],
      });
    }

    if (this.layouts.tonemapper && this.textures.outputTexture && this.sampler) {
      this.bindGroups.tonemapper = this.device.createBindGroup({
        layout: this.layouts.tonemapper,
        entries: [
          { binding: 0, resource: this.textures.outputTexture.createView() },
          { binding: 1, resource: this.sampler },
        ],
      });
    }
  }

  // ---------------- Morton helpers (CPU) ----------------

  _expandBits10(v) {
    v &= 1023;
    v = (v | (v << 16)) & 0x030000ff;
    v = (v | (v << 8)) & 0x0300f00f;
    v = (v | (v << 4)) & 0x030c30c3;
    v = (v | (v << 2)) & 0x09249249;
    return v >>> 0;
  }

  _morton3D(x, y, z) {
    const xx = this._expandBits10(x);
    const yy = this._expandBits10(y);
    const zz = this._expandBits10(z);
    return ((xx << 2) | (yy << 1) | zz) >>> 0;
  }

  buildMortonAndSort(trianglesData) {
    const numTris = (trianglesData.length / 9) | 0;
    if (numTris <= 0) {
      return { mortonSorted: new Uint32Array(0), triIndexSorted: new Uint32Array(0) };
    }

    let minX = 1e30, minY = 1e30, minZ = 1e30;
    let maxX = -1e30, maxY = -1e30, maxZ = -1e30;

    for (let t = 0; t < numTris; t++) {
      const b = t * 9;
      const cx = (trianglesData[b + 0] + trianglesData[b + 3] + trianglesData[b + 6]) / 3;
      const cy = (trianglesData[b + 1] + trianglesData[b + 4] + trianglesData[b + 7]) / 3;
      const cz = (trianglesData[b + 2] + trianglesData[b + 5] + trianglesData[b + 8]) / 3;

      minX = Math.min(minX, cx); minY = Math.min(minY, cy); minZ = Math.min(minZ, cz);
      maxX = Math.max(maxX, cx); maxY = Math.max(maxY, cy); maxZ = Math.max(maxZ, cz);
    }

    const dx = Math.max(1e-20, maxX - minX);
    const dy = Math.max(1e-20, maxY - minY);
    const dz = Math.max(1e-20, maxZ - minZ);

    const pairs = new Array(numTris);

    for (let t = 0; t < numTris; t++) {
      const b = t * 9;
      const cx = (trianglesData[b + 0] + trianglesData[b + 3] + trianglesData[b + 6]) / 3;
      const cy = (trianglesData[b + 1] + trianglesData[b + 4] + trianglesData[b + 7]) / 3;
      const cz = (trianglesData[b + 2] + trianglesData[b + 5] + trianglesData[b + 8]) / 3;

      const nx = (cx - minX) / dx;
      const ny = (cy - minY) / dy;
      const nz = (cz - minZ) / dz;

      const qx = Math.max(0, Math.min(1023, (nx * 1023) | 0));
      const qy = Math.max(0, Math.min(1023, (ny * 1023) | 0));
      const qz = Math.max(0, Math.min(1023, (nz * 1023) | 0));

      const code = this._morton3D(qx, qy, qz);
      pairs[t] = { code, tri: t };
    }

    pairs.sort((a, b) => (a.code - b.code) || (a.tri - b.tri));

    const mortonSorted = new Uint32Array(numTris);
    const triIndexSorted = new Uint32Array(numTris);

    for (let i = 0; i < numTris; i++) {
      mortonSorted[i] = pairs[i].code >>> 0;
      triIndexSorted[i] = pairs[i].tri >>> 0;
    }

    return { mortonSorted, triIndexSorted };
  }

  // ---------------- Readback LBVH2 ----------------

  async readBVH2(bytes) {
    const size = Math.max(4, bytes);

    const readBuffer = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const cmd = this.device.createCommandEncoder();
    cmd.copyBufferToBuffer(this.buffers.BVH2, 0, readBuffer, 0, size);
    this.device.queue.submit([cmd.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const arr = new Uint32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    readBuffer.destroy();
    return arr;
  }

  // ---------------- Collapse LBVH2 -> BVH4 (CPU) ----------------

  collapseLBVH2ToBVH4(bvh2U32, numTris) {
    const numNodes2 = (numTris > 0 ? (2 * numTris - 1) : 0);

    if (numNodes2 === 0) {
      return { bvh4U32: new Uint32Array([0]), numNodes4: 0 };
    }

    const node2Offset = (i) => 1 + i * NODE2_STRIDE_U32;

    const isLeaf2 = (i) => {
      const off = node2Offset(i);
      const meta = bvh2U32[off + 5] >>> 0;
      return (meta & LEAF_FLAG) !== 0;
    };

    const getChildren2 = (i) => {
      const off = node2Offset(i);
      return {
        left:  bvh2U32[off + 3] >>> 0,
        right: bvh2U32[off + 4] >>> 0,
      };
    };

    const getBoundsPacked2 = (i) => {
      const off = node2Offset(i);
      return [
        bvh2U32[off + 0] >>> 0,
        bvh2U32[off + 1] >>> 0,
        bvh2U32[off + 2] >>> 0,
      ];
    };

    const getMeta2 = (i) => {
      const off = node2Offset(i);
      return bvh2U32[off + 5] >>> 0;
    };

    // --- helpers (local, no external deps) ---

    const decodeBounds = (b0, b1, b2) => {
      // assumes your existing FP16 packing layout
      const mnx = unpack16x2(b0, 0);
      const mny = unpack16x2(b0, 1);
      const mnz = unpack16x2(b1, 0);
      const mxx = unpack16x2(b1, 1);
      const mxy = unpack16x2(b2, 0);
      const mxz = unpack16x2(b2, 1);

      return {
        min: [mnx, mny, mnz],
        max: [mxx, mxy, mxz],
      };
    };

    const encodeBounds = (min, max) => {
      return [
        pack16x2(min[0], min[1]),
        pack16x2(min[2], max[0]),
        pack16x2(max[1], max[2]),
      ];
    };

    // BVH4 builder (pre-order emission so root becomes node 0)
    const out = [];
    out.push(0); // placeholder for node count

    const emitNode4 = () => {
      const idx = ((out.length - 1) / NODE4_STRIDE_U32) | 0;
      for (let k = 0; k < NODE4_STRIDE_U32; k++) out.push(0);
      return idx;
    };

    const writeNode4 = (idx, b0, b1, b2, c0, c1, c2, c3, meta) => {
      const base = 1 + idx * NODE4_STRIDE_U32;
      out[base + 0] = b0 >>> 0;
      out[base + 1] = b1 >>> 0;
      out[base + 2] = b2 >>> 0;
      out[base + 3] = c0 >>> 0;
      out[base + 4] = c1 >>> 0;
      out[base + 5] = c2 >>> 0;
      out[base + 6] = c3 >>> 0;
      out[base + 7] = meta >>> 0;
    };

    const build4 = (node2) => {
      const idx4 = emitNode4();

      // ---- leaf passthrough ----
      if (isLeaf2(node2)) {
        const [b0, b1, b2] = getBoundsPacked2(node2);
        const meta2 = getMeta2(node2);
        writeNode4(idx4, b0, b1, b2,
                  INVALID, INVALID, INVALID, INVALID,
                  meta2);
        return idx4;
      }

      // ---- collect children ----
      const kids = [];
      const { left, right } = getChildren2(node2);
      kids.push(left, right);

      // greedy treelet collapse
      let changed = true;
      while (kids.length < 4 && changed) {
        changed = false;
        for (let i = 0; i < kids.length; i++) {
          const k = kids[i];
          if (k !== INVALID && !isLeaf2(k)) {
            const ch = getChildren2(k);
            kids.splice(i, 1, ch.left, ch.right);
            changed = true;
            break;
          }
        }
      }

      // ---- recurse children first ----
      const cIdx = [INVALID, INVALID, INVALID, INVALID];
      const boundsFP32 = {
        min: [ Infinity,  Infinity,  Infinity],
        max: [-Infinity, -Infinity, -Infinity],
      };

      for (let i = 0; i < 4; i++) {
        if (i < kids.length) {
          const ci = build4(kids[i]);
          cIdx[i] = ci;

          // merge child bounds in FP32
          const base = 1 + ci * NODE4_STRIDE_U32;
          const cb0 = out[base + 0];
          const cb1 = out[base + 1];
          const cb2 = out[base + 2];

          const b = decodeBounds(cb0, cb1, cb2);

          for (let k = 0; k < 3; k++) {
            boundsFP32.min[k] = Math.min(boundsFP32.min[k], b.min[k]);
            boundsFP32.max[k] = Math.max(boundsFP32.max[k], b.max[k]);
          }
        }
      }

      // ---- conservative expand (constant epsilon) ----
      const eps = BVH_EPSILON; // your existing constant
      for (let k = 0; k < 3; k++) {
        boundsFP32.min[k] -= eps;
        boundsFP32.max[k] += eps;
      }

      // ---- pack once ----
      const [b0, b1, b2] = encodeBounds(boundsFP32.min, boundsFP32.max);

      writeNode4(idx4, b0, b1, b2,
                cIdx[0], cIdx[1], cIdx[2], cIdx[3],
                0);

      return idx4;
    };

    // Root of LBVH2 is always node 0
    build4(0);

    const numNodes4 = ((out.length - 1) / NODE4_STRIDE_U32) | 0;
    out[0] = numNodes4 >>> 0;

    return { bvh4U32: new Uint32Array(out), numNodes4 };
  }

  // ---------------- Build BVH (LBVH4) ----------------

  async buildBVH(trianglesData) {
    const device = this.device;
    if (!device) return;

    let start = performance.now();

    const numTriangles = (trianglesData.length / 9) | 0;

    // upload triangles
    device.queue.writeBuffer(this.buffers.triangles, 0, trianglesData);

    // build Morton + sort on CPU
    const { mortonSorted, triIndexSorted } = this.buildMortonAndSort(trianglesData);

    // ensure BVH2 + aux sizes
    const { numNodes2, bytes: bvh2Bytes } = this.computeBVH2Sizing(numTriangles);
    this.ensureLBVHAuxBuffers(numTriangles, numNodes2);
    this.ensureBVH2Buffer(numTriangles);

    // upload morton + triIdx
    device.queue.writeBuffer(this.buffers.mortonSorted, 0, mortonSorted);
    device.queue.writeBuffer(this.buffers.triIndexSorted, 0, triIndexSorted);

    // builder UBO: x=numTris
    device.queue.writeBuffer(this.buffers.builderUBO, 0, new Uint32Array([numTriangles >>> 0, 0, 0, 0]));

    // BVH2[0] = numNodes2
    device.queue.writeBuffer(this.buffers.BVH2, 0, new Uint32Array([numNodes2 >>> 0]));

    if (numTriangles === 0) {
      // also ensure BVH4 is empty
      this.ensureBVH4Buffer(1);
      device.queue.writeBuffer(this.buffers.BVH, 0, new Uint32Array([0]));
      await device.queue.onSubmittedWorkDone();
      return;
    }

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, this.bindGroups.BVHBuilder);

    // internal pass
    if (numTriangles > 1) {
      pass.setPipeline(this.pipelines.BVHBuildInternal);
      pass.dispatchWorkgroups(Math.ceil((numTriangles - 1) / BVH_WORKGROUP_SIZE));
    }

    // leaves pass
    pass.setPipeline(this.pipelines.BVHBuildLeaves);
    pass.dispatchWorkgroups(Math.ceil(numTriangles / BVH_WORKGROUP_SIZE));

    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // read back BVH2 once (scene build time)
    const bvh2U32 = await this.readBVH2(bvh2Bytes);

    // collapse to BVH4
    const { bvh4U32, numNodes4 } = this.collapseLBVH2ToBVH4(bvh2U32, numTriangles);

    // upload BVH4
    this.ensureBVH4Buffer(Math.max(1, numNodes4));
    device.queue.writeBuffer(this.buffers.BVH, 0, bvh4U32);

    let end = performance.now();

    // done
    console.log("BVH Build Time:", end - start, "ms")
  }

  async setScene(scene) {
    this.trianglesData = scene.getTrianglesFloat32();
    await this.buildBVH(this.trianglesData);
  }

  async render() {
    if (!this.buffers.BVH) return;

    const numTriangles = (this.trianglesData.length / 9) | 0;

    let fov = (70.0 * Math.PI) / 180;
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

      this.frameCount,
      0,
      0,
      0,
    ]);

    this.device.queue.writeBuffer(this.buffers.rendererUBO, 0, UBO);

    const encoder = this.device.createCommandEncoder();

    // Compute render
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.renderer);
    pass.setBindGroup(0, this.bindGroups.renderer);

    const wgX = Math.ceil(this.canvas.width / 16);
    const wgY = Math.ceil(this.canvas.height / 16);
    pass.dispatchWorkgroups(wgX, wgY);

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

  setFrameCount(frameCount) {
    this.frameCount = frameCount;
  }
}
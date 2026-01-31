// PathTracer.js (full replacement)
// - NO shader merging / NO JS concatenation / NO WGSL #include required at runtime
// - JS *reads* BVH4Layout.wgsl only to size buffers consistently (optional but recommended)
// - BVH hierarchy is built on CPU (morton + radix + uplift), GPU builds bounds bottom-up level-by-level
//
// Expected WGSL files (each compiled independently):
//   ./src/shaders/BVH4Layout.wgsl        (ABI constants; JS reads it for sizing)
//   ./src/shaders/BVH4BoundsBuilder.wgsl (entrypoints: buildLeaves, buildParents)
//   ./src/shaders/renderer.wgsl
//   ./src/shaders/tonemapper.wgsl
//
// NOTE: BVH4BoundsBuilder.wgsl and renderer.wgsl must BOTH agree on the BVH buffer ABI,
// but they do NOT “interact” or share code. They just read/write the same raw buffers.

import * as io from "./io.js";

const MB = 1048576;
const BVH_WORKGROUP_SIZE = 256;
const INVALID_U32 = 0xFFFFFFFF >>> 0;

// ─────────────────────────────────────────────────────────────
// CPU helpers (morton + radix + hierarchy grouping)
// ─────────────────────────────────────────────────────────────

function clamp01(x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

// 10-bit morton expansion => 30-bit morton code
function expandBits10(v) {
  v &= 1023;
  v = (v | (v << 16)) & 0x030000ff;
  v = (v | (v << 8)) & 0x0300f00f;
  v = (v | (v << 4)) & 0x030c30c3;
  v = (v | (v << 2)) & 0x09249249;
  return v >>> 0;
}

function morton3D_10bit(x, y, z) {
  const xx = expandBits10(x);
  const yy = expandBits10(y);
  const zz = expandBits10(z);
  return (xx | (yy << 1) | (zz << 2)) >>> 0;
}

// LSD radix sort (8-bit passes) sorting (codes, items) by codes
function radixSortUint32ByKey(codes, items) {
  const n = codes.length;
  const tmpCodes = new Uint32Array(n);
  const tmpItems = new Uint32Array(n);
  const counts = new Uint32Array(256);

  for (let pass = 0; pass < 4; pass++) {
    counts.fill(0);
    const shift = pass * 8;

    for (let i = 0; i < n; i++) counts[(codes[i] >>> shift) & 255]++;

    // prefix sum
    let sum = 0;
    for (let k = 0; k < 256; k++) {
      const c = counts[k];
      counts[k] = sum;
      sum += c;
    }

    for (let i = 0; i < n; i++) {
      const key = (codes[i] >>> shift) & 255;
      const dst = counts[key]++;
      tmpCodes[dst] = codes[i];
      tmpItems[dst] = items[i];
    }

    codes.set(tmpCodes);
    items.set(tmpItems);
  }
}

// Compute leaf AABB centers (one leaf per triangle)
function computeLeafCenters(trisF32) {
  const n = (trisF32.length / 9) | 0;
  const centers = new Float32Array(n * 3);

  for (let i = 0; i < n; i++) {
    const b = i * 9;

    const x0 = trisF32[b + 0], y0 = trisF32[b + 1], z0 = trisF32[b + 2];
    const x1 = trisF32[b + 3], y1 = trisF32[b + 4], z1 = trisF32[b + 5];
    const x2 = trisF32[b + 6], y2 = trisF32[b + 7], z2 = trisF32[b + 8];

    const mnx = Math.min(x0, x1, x2);
    const mny = Math.min(y0, y1, y2);
    const mnz = Math.min(z0, z1, z2);
    const mxx = Math.max(x0, x1, x2);
    const mxy = Math.max(y0, y1, y2);
    const mxz = Math.max(z0, z1, z2);

    centers[i * 3 + 0] = 0.5 * (mnx + mxx);
    centers[i * 3 + 1] = 0.5 * (mny + mxy);
    centers[i * 3 + 2] = 0.5 * (mnz + mxz);
  }

  return centers;
}

function mortonSortNodes(activeNodeIndices, nodeCenters3f) {
  const n = activeNodeIndices.length;
  const codes = new Uint32Array(n);
  const items = new Uint32Array(n);

  // bounds of active centers for normalization
  let mnx = 1e30, mny = 1e30, mnz = 1e30;
  let mxx = -1e30, mxy = -1e30, mxz = -1e30;

  for (let i = 0; i < n; i++) {
    const node = activeNodeIndices[i];
    const c = node * 3;
    const x = nodeCenters3f[c + 0];
    const y = nodeCenters3f[c + 1];
    const z = nodeCenters3f[c + 2];
    if (x < mnx) mnx = x; if (x > mxx) mxx = x;
    if (y < mny) mny = y; if (y > mxy) mxy = y;
    if (z < mnz) mnz = z; if (z > mxz) mxz = z;
  }

  const ex = (mxx - mnx) || 1.0;
  const ey = (mxy - mny) || 1.0;
  const ez = (mxz - mnz) || 1.0;

  for (let i = 0; i < n; i++) {
    const node = activeNodeIndices[i];
    const c = node * 3;

    const nx = clamp01((nodeCenters3f[c + 0] - mnx) / ex);
    const ny = clamp01((nodeCenters3f[c + 1] - mny) / ey);
    const nz = clamp01((nodeCenters3f[c + 2] - mnz) / ez);

    const ix = (nx * 1023) | 0;
    const iy = (ny * 1023) | 0;
    const iz = (nz * 1023) | 0;

    codes[i] = morton3D_10bit(ix, iy, iz);
    items[i] = node >>> 0;
  }

  radixSortUint32ByKey(codes, items);
  return items; // sorted node indices
}

// CPU hierarchy build producing:
// - totalNodes, rootIndex
// - levels[] describing parents batches (baseNode, groupOffsetU32, groupsCount)
// - groupsU32 : flattened child indices (4 per parent), INVALID_U32 for missing
function buildBVH4HierarchyBottomUp(leafCenters3f, numLeaves) {
  const maxNodes = Math.max(1, 2 * numLeaves - 1);
  const nodeCenters3f = new Float32Array(maxNodes * 3);
  nodeCenters3f.set(leafCenters3f, 0);

  let nextNodeIndex = numLeaves;

  let active = new Uint32Array(numLeaves);
  for (let i = 0; i < numLeaves; i++) active[i] = i;

  const levels = [];
  const groupsList = [];

  while (active.length > 1) {
    const sorted = mortonSortNodes(active, nodeCenters3f);

    let len = sorted.length;
    let upliftNode = INVALID_U32;

    // remainder rule:
    // - rem 2/3 => group them
    // - rem 1   => uplift last node to next level
    const rem = len & 3;
    if (rem === 1) {
      upliftNode = sorted[len - 1];
      len --;
    }

    const baseNode = nextNodeIndex;
    const groupOffsetU32 = groupsList.length;
    let parentsCreated = 0;

    let i = 0;
    while (i < len) {
      const remaining = len - i;
      const take = remaining >= 4 ? 4 : remaining; // 2 or 3 possible at end

      const c0 = sorted[i + 0];
      const c1 = take > 1 ? sorted[i + 1] : INVALID_U32;
      const c2 = take > 2 ? sorted[i + 2] : INVALID_U32;
      const c3 = take > 3 ? sorted[i + 3] : INVALID_U32;

      groupsList.push(c0, c1, c2, c3);

      const p = nextNodeIndex++;
      parentsCreated++;

      // parent center = average(child centers)
      let sx = 0, sy = 0, sz = 0, cnt = 0;

      const kids = [c0, c1, c2, c3];
      for (let k = 0; k < 4; k++) {
        const child = kids[k];
        if (child !== INVALID_U32) {
          const cc = child * 3;
          sx += nodeCenters3f[cc + 0];
          sy += nodeCenters3f[cc + 1];
          sz += nodeCenters3f[cc + 2];
          cnt++;
        }
      }

      const pc = p * 3;
      nodeCenters3f[pc + 0] = sx / cnt;
      nodeCenters3f[pc + 1] = sy / cnt;
      nodeCenters3f[pc + 2] = sz / cnt;

      i += take;
    }

    levels.push({ baseNode, groupOffsetU32, groupsCount: parentsCreated });

    // next active = parents + optional uplift
    const nextActiveLen = parentsCreated + (upliftNode !== INVALID_U32 ? 1 : 0);
    const nextActive = new Uint32Array(nextActiveLen);

    for (let k = 0; k < parentsCreated; k++) nextActive[k] = (baseNode + k) >>> 0;
    if (upliftNode !== INVALID_U32) nextActive[nextActiveLen - 1] = upliftNode >>> 0;

    active = nextActive;
  }

  const totalNodes = nextNodeIndex;
  const rootIndex = active[0] >>> 0;

  return {
    totalNodes,
    rootIndex,
    levels,
    groupsU32: new Uint32Array(groupsList),
  };
}

function bytesPerPixelForFormat(format) {
  switch (format) {
    case "rgba8unorm":
    case "rgba8snorm":
    case "rgba8uint":
    case "rgba8sint":
      return 4;

    case "bgra8unorm":
      return 4;

    case "rgba16float":
      return 8;

    case "rgba32float":
      return 16;

    case "r32float":
      return 4;

    default:
      console.warn("Unknown texture format:", format);
      return 0;
  }
}

function estimateTextureBytes(texture, descriptor) {
  const { size, format, mipLevelCount = 1 } = descriptor;

  const width  = size[0];
  const height = size[1];
  const layers = size[2] ?? 1;

  const bpp = bytesPerPixelForFormat(format);
  let total = 0;

  let w = width;
  let h = height;

  for (let mip = 0; mip < mipLevelCount; mip++) {
    total += w * h * layers * bpp;
    w = Math.max(1, w >> 1);
    h = Math.max(1, h >> 1);
  }

  return total;
}


// ─────────────────────────────────────────────────────────────
// PathTracer
// ─────────────────────────────────────────────────────────────

export class PathTracer {
  constructor(canvas) {
    this.canvas = canvas;

    this.adapter = null;
    this.device = null;
    this.canvasContext = null;

    // Camera
    this.cameraPosition = [0.0, 0.0, 3.5];
    this.cameraQuaternion = [0.0, 0.0, 0.0, 1.0];

    this.frameCount = 0;

    // BVH layout defaults (overridden by BVH4Layout.wgsl if parsed)
    this.bvhLayout = {
      headerWordsU32: 2,      // [totalNodes, rootIndex]
      nodeStrideU32: 4,       // packed bounds/meta per node (example)
      childStrideU32: 4,      // 4 children per node
    };

    this.buffers = {};
    this.textures = {};
    this.layouts = {};
    this.pipelines = {};
    this.bindGroups = {};

    // default mesh (4 triangles)
    this.trianglesData = new Float32Array([
      1, 1, 1, -1, -1, 1, -1, 1, -1,
      1, 1, 1, -1, 1, -1, 1, -1, -1,
      1, 1, 1, 1, -1, -1, -1, -1, 1,
      -1, -1, 1, 1, -1, -1, -1, 1, -1,
    ]);

    // Cached BVH info
    this.bvhTotalNodes = 0;
    this.bvhRootIndex = 0;
  }

  async initialize() {
    await this.getDeviceAndContext();
    await this.loadShadersAndLayout();
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

  // Read BVH4Layout.wgsl for ABI constants (NO merging; just parsing constants for sizing).
  async loadBVHLayoutConstants(path) {
    const txt = await io.loadText(path);

    const readU32Const = (names, fallback) => {
      for (const name of names) {
        // matches: const NAME : u32 = 4u;
        const re = new RegExp(`const\\s+${name}\\s*:\\s*u32\\s*=\\s*([0-9]+)u\\s*;`);
        const m = txt.match(re);
        if (m) return (parseInt(m[1], 10) >>> 0);
      }
      return fallback >>> 0;
    };

    // You can rename these constants in BVH4Layout.wgsl—just add aliases here.
    const headerWordsU32 = readU32Const(
      ["BVH4_HEADER_WORDS", "BVH_HEADER_WORDS", "BVH_HEADER_U32_WORDS"],
      this.bvhLayout.headerWordsU32
    );

    const nodeStrideU32 = readU32Const(
      ["BVH4_NODE_STRIDE_U32", "BVH_NODE_STRIDE_U32"],
      this.bvhLayout.nodeStrideU32
    );

    const childStrideU32 = readU32Const(
      ["BVH4_CHILD_STRIDE_U32", "BVH_CHILD_STRIDE_U32"],
      this.bvhLayout.childStrideU32
    );

    this.bvhLayout = { headerWordsU32, nodeStrideU32, childStrideU32 };
  }

  async loadShadersAndLayout() {
    // BVH layout (optional parse for ABI sizing)
    // This does NOT make shaders “interact”. It’s just JS reading constants.
    try {
      await this.loadBVHLayoutConstants("./src/shaders/BVHLayout.wgsl");
      console.log("BVH4Layout constants:", this.bvhLayout);
    } catch (e) {
      console.warn("Could not parse BVH4Layout.wgsl constants; using defaults.", e);
    }

    const BVH4BoundsBuilderCode = await io.loadText("./src/shaders/BVHBuilder.wgsl");
    const rendererCode = await io.loadText("./src/shaders/renderer.wgsl");
    const tonemapperCode = await io.loadText("./src/shaders/tonemapper.wgsl");

    this.shaders = {
      BVH4BoundsBuilder: { shader: this.device.createShaderModule({ code: BVH4BoundsBuilderCode }) },
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
        size: 4 * MB, // resized as needed
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      }),

      BVH: null,         // bounds + meta (u32)
      BVHChildren: null, // explicit children (u32)
      BVHGroups: null,   // flattened groups for all levels (u32)
    };

    // minimal placeholders
    this.ensureBVHBuffers(1, 4);

    this.textures = {
      outputTexture: this.device.createTexture({
        size: [width, height],
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING,
      }),
      descriptor: {
        size: [width, height],
        format: "rgba8unorm",
        mipLevelCount: 1,
      },
    };
  }

  ensureTrianglesBuffer(bytesNeeded) {
    if (this.buffers.triangles.size >= bytesNeeded) return;

    console.log(`Realloc triangles buffer: ${(bytesNeeded / MB).toFixed(2)} MB`);
    this.buffers.triangles.destroy();
    this.buffers.triangles = this.device.createBuffer({
      size: bytesNeeded,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.updateBindGroups();
  }

  ensureBVHBuffers(totalNodes, groupsU32Length) {
    const { headerWordsU32, nodeStrideU32, childStrideU32 } = this.bvhLayout;

    const bvhU32 = headerWordsU32 + totalNodes * nodeStrideU32;
    const bvhBytes = Math.max(4, bvhU32 * 4);

    const childrenU32 = totalNodes * childStrideU32;
    const childrenBytes = Math.max(4, childrenU32 * 4);

    const groupsBytes = Math.max(4, groupsU32Length * 4);

    // BVH
    if (!this.buffers.BVH || this.buffers.BVH.size < bvhBytes) {
      if (this.buffers.BVH) this.buffers.BVH.destroy();
      this.buffers.BVH = this.device.createBuffer({
        size: bvhBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    }

    // children
    if (!this.buffers.BVHChildren || this.buffers.BVHChildren.size < childrenBytes) {
      if (this.buffers.BVHChildren) this.buffers.BVHChildren.destroy();
      this.buffers.BVHChildren = this.device.createBuffer({
        size: childrenBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    }

    // groups
    if (!this.buffers.BVHGroups || this.buffers.BVHGroups.size < groupsBytes) {
      if (this.buffers.BVHGroups) this.buffers.BVHGroups.destroy();
      this.buffers.BVHGroups = this.device.createBuffer({
        size: groupsBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }

    this.updateBindGroups();
  }

  async createBindGroupsAndPipelines() {
    const device = this.device;

    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    // Builder bindgroup is separate from renderer bindgroup.
    // No cross-use. No “interaction”. Just separate pipelines over shared buffers.
    this.layouts = {
      bvhBoundsBuilder: device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // BVH (rw)
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // triangles
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, // builder UBO
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // BVHChildren (rw)
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // BVHGroups
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
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, // renderer UBO
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // triangles
          { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // BVH
          { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // BVHChildren
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
      BVHLeaves: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.layouts.bvhBoundsBuilder] }),
        compute: { module: this.shaders.BVH4BoundsBuilder.shader, entryPoint: "buildLeaves" },
      }),

      BVHParents: device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.layouts.bvhBoundsBuilder] }),
        compute: { module: this.shaders.BVH4BoundsBuilder.shader, entryPoint: "buildParents" },
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
    if (!this.layouts || !this.layouts.bvhBoundsBuilder) return;
    if (!this.buffers.BVH || !this.buffers.BVHChildren || !this.buffers.BVHGroups) return;

    this.bindGroups.BVHBoundsBuilder = this.device.createBindGroup({
      layout: this.layouts.bvhBoundsBuilder,
      entries: [
        { binding: 0, resource: { buffer: this.buffers.BVH } },
        { binding: 1, resource: { buffer: this.buffers.triangles } },
        { binding: 2, resource: { buffer: this.buffers.builderUBO } },
        { binding: 3, resource: { buffer: this.buffers.BVHChildren } },
        { binding: 4, resource: { buffer: this.buffers.BVHGroups } },
      ],
    });

    if (this.textures.outputTexture && this.layouts.renderer) {
      this.bindGroups.renderer = this.device.createBindGroup({
        layout: this.layouts.renderer,
        entries: [
          { binding: 0, resource: this.textures.outputTexture.createView() },
          { binding: 1, resource: { buffer: this.buffers.rendererUBO } },
          { binding: 2, resource: { buffer: this.buffers.triangles } },
          { binding: 3, resource: { buffer: this.buffers.BVH } },
          { binding: 4, resource: { buffer: this.buffers.BVHChildren } },
        ],
      });
    }

    if (this.textures.outputTexture && this.layouts.tonemapper && this.sampler) {
      this.bindGroups.tonemapper = this.device.createBindGroup({
        layout: this.layouts.tonemapper,
        entries: [
          { binding: 0, resource: this.textures.outputTexture.createView() },
          { binding: 1, resource: this.sampler },
        ],
      });
    }
  }

  async buildBVH(trianglesData) {
    const device = this.device;
    const start = performance.now();

    const trisF32 = trianglesData instanceof Float32Array ? trianglesData : new Float32Array(trianglesData);
    const numTriangles = (trisF32.length / 9) | 0;

    if (numTriangles === 0) {
      console.warn("No triangles to build BVH for.");
      return;
    }

    // Ensure triangles buffer
    const triBytes = trisF32.byteLength;
    this.ensureTrianglesBuffer(triBytes);

    // CPU hierarchy (fast + deterministic)
    const cpu0 = performance.now();
    const leafCenters = computeLeafCenters(trisF32);
    const hierarchy = buildBVH4HierarchyBottomUp(leafCenters, numTriangles);
    const { totalNodes, rootIndex, levels, groupsU32 } = hierarchy;
    const cpu1 = performance.now();

    // Allocate BVH buffers per layout
    this.ensureBVHBuffers(totalNodes, groupsU32.length);

    // Upload stage
    const up0 = performance.now();

    device.queue.writeBuffer(this.buffers.triangles, 0, trisF32);

    // BVH header at offset 0:
    // [totalNodes, rootIndex] in u32
    device.queue.writeBuffer(this.buffers.BVH, 0, new Uint32Array([totalNodes >>> 0, rootIndex >>> 0]));

    if (groupsU32.length > 0) device.queue.writeBuffer(this.buffers.BVHGroups, 0, groupsU32);

    const up1 = performance.now();

    // GPU stage: leaves then parent levels
    const gpu0 = performance.now();

    // Pass 1: build leaves
    {
      // builderUBO meaning is up to your BVH4BoundsBuilder.wgsl
      // Convention here:
      //   buildLeaves: ubo = [numLeaves (=numTriangles), 0, 0, 0]
      device.queue.writeBuffer(this.buffers.builderUBO, 0, new Uint32Array([numTriangles, 0, 0, 0]));

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.BVHLeaves);
      pass.setBindGroup(0, this.bindGroups.BVHBoundsBuilder);
      pass.dispatchWorkgroups(Math.ceil(numTriangles / BVH_WORKGROUP_SIZE));
      pass.end();
      device.queue.submit([encoder.finish()]);
    }

    // Pass 2..N: build parents per level
    for (let li = 0; li < levels.length; li++) {
      const L = levels[li];

      // Convention:
      //   buildParents: ubo = [numLeaves, baseNode, groupsOffsetU32, groupsCount]
      device.queue.writeBuffer(
        this.buffers.builderUBO,
        0,
        new Uint32Array([numTriangles, L.baseNode >>> 0, L.groupOffsetU32 >>> 0, L.groupsCount >>> 0])
      );

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.BVHParents);
      pass.setBindGroup(0, this.bindGroups.BVHBoundsBuilder);
      pass.dispatchWorkgroups(Math.ceil(L.groupsCount / BVH_WORKGROUP_SIZE));
      pass.end();
      device.queue.submit([encoder.finish()]);
    }

    await device.queue.onSubmittedWorkDone();

    const gpu1 = performance.now();
    const end = performance.now();

    this.bvhTotalNodes = totalNodes;
    this.bvhRootIndex = rootIndex;

    console.log(
      `BVH4 build: leaves=${numTriangles}, totalNodes=${totalNodes}, levels=${levels.length}, root=${rootIndex}`
    );
    console.log(`BVH build complete: ${(end - start).toFixed(2)} ms`);
    console.log(`GPU Part: ${(gpu1 - gpu0).toFixed(2)} ms`);
    console.log(`CPU Part: ${(cpu1 - cpu0).toFixed(2)} ms`);
    console.log(`PCIe Part: ${(up1 - up0).toFixed(2)} ms`);
  }

  async setScene(scene) {
    this.trianglesData = scene.getTrianglesFloat32();
    await this.buildBVH(this.trianglesData);
  }

  async render() {
    if (!this.bindGroups.renderer) return;

    const numTriangles = (this.trianglesData.length / 9) | 0;

    const fov = (70.0 * Math.PI) / 180.0;
    const focal = 1.0 / Math.tan(0.5 * fov);

    // Renderer UBO matches your existing renderer.wgsl expectations
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

    // Path trace compute
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.renderer);
      pass.setBindGroup(0, this.bindGroups.renderer);
      const wgX = Math.ceil(this.canvas.width / 16);
      const wgY = Math.ceil(this.canvas.height / 16);
      pass.dispatchWorkgroups(wgX, wgY);
      pass.end();
    }

    // Tonemap to swapchain
    {
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
    }

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

  // Optional debug: read BVH u32s back
  async readBVH_u32() {
    const size = this.buffers.BVH.size;

    const readBuffer = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const cmd = this.device.createCommandEncoder();
    cmd.copyBufferToBuffer(this.buffers.BVH, 0, readBuffer, 0, size);
    this.device.queue.submit([cmd.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);
    const arr = new Uint32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    return arr;
  }

  estimateVRAMUsage() {
    let totalBytes = 0;
    const breakdown = [];

    // ───────── Buffers ─────────
    for (const [name, buf] of Object.entries(this.buffers)) {
      if (!buf || typeof buf.size !== "number") continue;
      totalBytes += buf.size;
      breakdown.push({ type: "buffer", name, bytes: buf.size });
    }

    // ───────── Textures ─────────
    // You MUST store texture descriptors if you want exact sizing
    // (WebGPU does not expose them later)
    for (const [name, texInfo] of Object.entries(this.textures)) {
      if (!texInfo || !texInfo.descriptor) continue;

      const bytes = estimateTextureBytes(texInfo.texture, texInfo.descriptor);
      totalBytes += bytes;
      breakdown.push({ type: "texture", name, bytes });
    }

    return JSON.stringify({
      totalBytes,
      totalMB: (totalBytes / (1024 * 1024)).toFixed(2),
      breakdown,
    });
  }
}
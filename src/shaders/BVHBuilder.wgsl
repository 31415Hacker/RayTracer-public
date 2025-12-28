// BVHBuilder.wgsl
// Builds an LBVH2 (binary) on GPU from sorted Morton codes.
// The CPU will collapse LBVH2 -> BVH4 after reading this buffer once.

const NODE2_STRIDE_U32: u32 = 6u;
const LEAF_FLAG: u32 = 0x80000000u;
const INVALID: u32 = 0xFFFFFFFFu;

@group(0) @binding(0)
var<storage, read_write> BVH2: array<u32>;

@group(0) @binding(1)
var<storage, read> triangles: array<f32>;

// Morton codes in sorted order (length = numTris)
@group(0) @binding(2)
var<storage, read> mortonSorted: array<u32>;

// Triangle indices in sorted order (length = numTris)
@group(0) @binding(3)
var<storage, read> triIndexSorted: array<u32>;

// Parent pointer for every LBVH2 node (length = numNodes2)
@group(0) @binding(4)
var<storage, read_write> parent: array<u32>;

// Internal-node flags (length = max(numTris-1, 1))
// Each internal node waits for 2 children; second arrival computes its AABB.
@group(0) @binding(5)
var<storage, read_write> buildFlags: array<atomic<u32>>;

// ubo.x = numTris
@group(0) @binding(6)
var<uniform> ubo: vec4<u32>;

fn getTriangleBoundsByTriIndex(triIndex: u32) -> array<vec3<f32>, 2u> {
    let base = triIndex * 9u;

    let v0 = vec3<f32>(
        triangles[base + 0u],
        triangles[base + 1u],
        triangles[base + 2u]
    );
    let v1 = vec3<f32>(
        triangles[base + 3u],
        triangles[base + 4u],
        triangles[base + 5u]
    );
    let v2 = vec3<f32>(
        triangles[base + 6u],
        triangles[base + 7u],
        triangles[base + 8u]
    );

    let mn = min(v0, min(v1, v2));
    let mx = max(v0, max(v1, v2));
    return array<vec3<f32>, 2u>(mn, mx);
}

// Treat an f32 as if it were stored in fp16, and move it by 1 fp16 ULP.
// up=true  -> next representable fp16 value
// up=false -> previous representable fp16 value
fn incrementF16(value: f32, up: bool, iterations: u32) -> f32 {
    // Pack value into lower 16 bits as fp16
    let p: u32 = pack2x16float(vec2<f32>(value, 0.0));
    let bits: u32 = p & 0xFFFFu;

    // Map fp16 bits -> ordered u16 space (monotonic for numeric ordering)
    let sign: bool = (bits & 0x8000u) != 0u;
    var ord: u32 = select(bits ^ 0x8000u, (~bits) & 0xFFFFu, sign);

    // Step in ordered space
    ord = select(ord - iterations, ord + iterations, up);

    // Map ordered u16 -> fp16 bits
    let ordSign: bool = (ord & 0x8000u) != 0u;
    let bits2: u32 = select((~ord) & 0xFFFFu, ord ^ 0x8000u, ordSign);

    // Unpack back to f32 (from fp16)
    return unpack2x16float(bits2).x;
}

fn writeBounds2(node: u32, mn: vec3<f32>, mx: vec3<f32>) {
    let base: u32 = 1u + node * NODE2_STRIDE_U32;

    // Expand by exactly 1 fp16 ULP in each component: mn down, mx up
    let mnL: vec3<f32> = vec3<f32>(
        incrementF16(mn.x, false, 1u),
        incrementF16(mn.y, false, 1u),
        incrementF16(mn.z, false, 1u)
    );

    let mxL: vec3<f32> = vec3<f32>(
        incrementF16(mx.x, true, 1u),
        incrementF16(mx.y, true, 1u),
        incrementF16(mx.z, true, 1u)
    );

    BVH2[base + 0u] = pack2x16float(vec2<f32>(mnL.x, mnL.y));
    BVH2[base + 1u] = pack2x16float(vec2<f32>(mnL.z, mxL.x));
    BVH2[base + 2u] = pack2x16float(vec2<f32>(mxL.y, mxL.z));
}

fn readBounds2(node: u32) -> array<vec3<f32>, 2u> {
    let base = 1u + node * NODE2_STRIDE_U32;

    let a = unpack2x16float(BVH2[base + 0u]);
    let b = unpack2x16float(BVH2[base + 1u]);
    let c = unpack2x16float(BVH2[base + 2u]);

    let mn = vec3<f32>(a.x, a.y, b.x);
    let mx = vec3<f32>(b.y, c.x, c.y);
    return array<vec3<f32>, 2u>(mn, mx);
}

fn writeInternal2(node: u32, left: u32, right: u32) {
    let base = 1u + node * NODE2_STRIDE_U32;

    BVH2[base + 3u] = left;
    BVH2[base + 4u] = right;
    BVH2[base + 5u] = 0u;
}

fn writeLeaf2(node: u32, triIndex: u32, mn: vec3<f32>, mx: vec3<f32>) {
    let base = 1u + node * NODE2_STRIDE_U32;

    writeBounds2(node, mn, mx);

    BVH2[base + 3u] = 0u;
    BVH2[base + 4u] = 0u;
    BVH2[base + 5u] = LEAF_FLAG | (triIndex & 0x7FFFFFFFu);
}

fn delta(i: i32, j: i32, n: i32) -> i32 {
    if (j < 0 || j >= n) {
        return -1;
    }

    let a = mortonSorted[u32(i)];
    let b = mortonSorted[u32(j)];
    let x = a ^ b;

    if (x == 0u) {
        let y = u32(i) ^ u32(j);
        return 32 + i32(countLeadingZeros(y));
    }

    return i32(countLeadingZeros(x));
}

// ---- Pass 1: connectivity (internal nodes) ----
@compute @workgroup_size(256)
fn buildInternal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let numTris: u32 = ubo.x;

    if (numTris <= 1u) {
        return;
    }

    let n: i32 = i32(numTris);
    let internalCount: u32 = numTris - 1u;

    let iU: u32 = gid.x;
    if (iU >= internalCount) {
        return;
    }

    let i: i32 = i32(iU);

    atomicStore(&buildFlags[iU], 0u);

    let dLeft = delta(i, i - 1, n);
    let dRight = delta(i, i + 1, n);
    let d: i32 = select(-1, 1, (dRight - dLeft) > 0);

    let deltaMin = delta(i, i - d, n);

    var lmax: i32 = 2;
    loop {
        if (delta(i, i + lmax * d, n) <= deltaMin) { break; }
        lmax = lmax << 1;
    }

    var l: i32 = 0;
    var t: i32 = lmax >> 1;

    loop {
        if (t <= 0) { break; }
        if (delta(i, i + (l + t) * d, n) > deltaMin) {
            l = l + t;
        }
        t = t >> 1;
    }

    let j: i32 = i + l * d;
    let first: i32 = min(i, j);
    let last: i32 = max(i, j);

    let deltaNode = delta(first, last, n);

    var split: i32 = first;
    var step: i32 = last - first;

    loop {
        if (step <= 1) { break; }
        step = (step + 1) >> 1;

        let newSplit = split + step;
        if (newSplit < last) {
            let dSplit = delta(first, newSplit, n);
            if (dSplit > deltaNode) {
                split = newSplit;
            }
        }
    }

    let leafBase: u32 = internalCount;

    let leftChild: u32 = select(
        u32(split),
        leafBase + u32(split),
        split == first
    );

    let rightIndex: i32 = split + 1;
    let rightChild: u32 = select(
        u32(rightIndex),
        leafBase + u32(rightIndex),
        rightIndex == last
    );

    writeInternal2(iU, leftChild, rightChild);

    parent[leftChild] = iU;
    parent[rightChild] = iU;

    if (iU == 0u) {
        parent[0u] = INVALID;
    }
}

fn propagateUp(fromNode: u32, internalCount: u32) {
    var node: u32 = fromNode;

    loop {
        let p: u32 = parent[node];
        if (p == INVALID) {
            break;
        }
        if (p >= internalCount) {
            break;
        }

        // First child arrives -> old == 0, stop.
        // Second child arrives -> old == 1, compute parent and keep going upward.
        let old: u32 = atomicAdd(&buildFlags[p], 1u);
        if (old == 0u) {
            break;
        }

        let pBase = 1u + p * NODE2_STRIDE_U32;
        let left  = BVH2[pBase + 3u];
        let right = BVH2[pBase + 4u];

        let lb = readBounds2(left);
        let rb = readBounds2(right);

        let mn = min(lb[0u], rb[0u]);
        let mx = max(lb[1u], rb[1u]);

        writeBounds2(p, mn, mx);

        node = p;
    }
}

// ---- Pass 2: leaves + bottom-up bounds ----
@compute @workgroup_size(256)
fn buildLeaves(@builtin(global_invocation_id) gid: vec3<u32>) {
    let numTris: u32 = ubo.x;

    if (numTris == 0u) {
        return;
    }

    let leafId: u32 = gid.x;
    if (leafId >= numTris) {
        return;
    }

    let internalCount: u32 = select(0u, numTris - 1u, numTris > 0u);
    let leafBase: u32 = internalCount;

    let nodeIndex: u32 = leafBase + leafId;

    let triIndex: u32 = triIndexSorted[leafId];
    let ab = getTriangleBoundsByTriIndex(triIndex);

    writeLeaf2(nodeIndex, triIndex, ab[0u], ab[1u]);

    if (internalCount > 0u) {
        propagateUp(nodeIndex, internalCount);
    } else {
        parent[0u] = INVALID;
    }
}
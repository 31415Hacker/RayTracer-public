// ─────────────────────────────────────────────
// BVH8 Builder with explicit leaf triangle ranges
// Layout in bvh buffer:
//
//   bvh[0] = totalNodes (as f32)
//   For node i:
//     base = 1u + i * 8u
//     [base + 0..2] = min.xyz
//     [base + 3..5] = max.xyz
//     [base + 6]    = f32(firstTri)
//     [base + 7]    = f32(triCount)
//
// Children are implicit in perfect 8-ary heap layout:
//   children of node i are at indices i*8+1 .. i*8+8
//   totalNodes = (8^(maxDepth+1) - 1) / 7
// Leaves are at depth = maxDepth.
// Triangles are evenly mapped onto leaves in contiguous ranges.
// ─────────────────────────────────────────────

@group(0) @binding(0)
var<storage, read_write> bvh : array<f32>;

@group(0) @binding(1)
var<storage, read> triangles : array<f32>;

@group(0) @binding(2)
var<uniform> ubo : vec4<u32>; // x = numTris, y = maxDepth, z,w unused

// -----------------------------------------
// Helpers
// -----------------------------------------

fn pow8(exp: u32) -> u32 {
    var r = 1u;
    for (var i = 0u; i < exp; i = i + 1u) {
        r = r * 8u;
    }
    return r;
}

fn getTriangleBounds(ti: u32) -> array<vec3<f32>, 2u> {
    let base = ti * 9u;

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

// Heap-style depth for 8-ary tree
fn nodeDepth(node: u32) -> u32 {
    if (node == 0u) {
        return 0u;
    }
    let val = f32(7u * node + 1u);
    let log8 = log2(val) / 3.0;
    return u32(floor(log8));
}

// Map a node index → [startTri, endTri) using a perfect 8-ary tree
// with leaves at depth = maxDepth, triangles distributed contiguously.
fn getNodeTriRange(node: u32, numTris: u32, maxDepth: u32) -> vec2<u32> {
    let depth = nodeDepth(node);
    if (depth > maxDepth) {
        return vec2<u32>(0u, 0u);
    }

    // First node index at this depth
    let startNodeAtDepth = (pow8(depth) - 1u) / 7u;
    let nodeOffset = node - startNodeAtDepth;

    let totalLeaves   = pow8(maxDepth);
    let leavesPerNode = pow8(maxDepth - depth);

    var leafStartIdx = nodeOffset * leavesPerNode;
    var leafEndIdx   = leafStartIdx + leavesPerNode;

    if (leafStartIdx > totalLeaves) {
        leafStartIdx = totalLeaves;
    }
    if (leafEndIdx > totalLeaves) {
        leafEndIdx = totalLeaves;
    }

    // Evenly distribute triangles over leaves; leaves beyond numTris are empty
    var startTri = leafStartIdx;
    var endTri   = leafEndIdx;

    if (startTri > numTris) {
        startTri = numTris;
    }
    if (endTri > numTris) {
        endTri = numTris;
    }

    return vec2<u32>(startTri, endTri);
}

// Write one node into the flattened BVH buffer
fn writeNode(
    i: u32,
    mn: vec3<f32>,
    mx: vec3<f32>,
    firstTri: u32,
    triCount: u32
) {
    let base = 1u + i * 8u;

    bvh[base + 0u] = mn.x;
    bvh[base + 1u] = mn.y;
    bvh[base + 2u] = mn.z;

    bvh[base + 3u] = mx.x;
    bvh[base + 4u] = mx.y;
    bvh[base + 5u] = mx.z;

    bvh[base + 6u] = f32(firstTri);
    bvh[base + 7u] = f32(triCount);
}

// -----------------------------------------
// Main builder
// -----------------------------------------
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let numTris  = ubo.x;
    let maxDepth = ubo.y;

    // Perfect 8-ary heap node count: 1 + 8 + 8^2 + ... + 8^maxDepth
    let totalNodes = (pow8(maxDepth + 1u) - 1u) / 7u;
    let node = gid.x;

    if (node >= totalNodes) {
        return;
    }

    // First thread writes metadata once: total node count
    if (gid.x == 0u) {
        bvh[0u] = f32(totalNodes);
    }

    let triRange = getNodeTriRange(node, numTris, maxDepth);
    var startTri = triRange.x;
    var endTri   = triRange.y;

    // Empty node → degenerate box and zero tri range
    if (startTri >= endTri) {
        writeNode(node, vec3<f32>(1e30), vec3<f32>(-1e30), 0u, 0u);
        return;
    }

    // Accumulate bounds over this node's triangle range
    var mn = vec3<f32>( 1e30);
    var mx = vec3<f32>(-1e30);

    var ti = startTri;
    loop {
        if (ti >= endTri) {
            break;
        }

        let ab = getTriangleBounds(ti);
        mn = min(mn, ab[0u]);
        mx = max(mx, ab[1u]);

        ti = ti + 1u;
    }

    // Leaf nodes (depth == maxDepth) actually own the triangle range
    let depth = nodeDepth(node);
    var firstTriLeaf: u32 = 0u;
    var triCountLeaf: u32 = 0u;

    if (depth == maxDepth) {
        firstTriLeaf = startTri;
        triCountLeaf = endTri - startTri;
    }

    writeNode(node, mn, mx, firstTriLeaf, triCountLeaf);
}
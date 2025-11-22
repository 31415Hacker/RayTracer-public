// ─────────────────────────────────────────────
// BVH8 Builder (Implicit Heap Layout)
// Children of node i: 8*i+1 ... 8*i+8
// ─────────────────────────────────────────────

@group(0) @binding(0)
var<storage, read_write> bvh : array<f32>;    // [nodeMin.xyz, nodeMax.xyz]

@group(0) @binding(1)
var<storage, read> triangles : array<f32>;    // 9 floats per tri

@group(0) @binding(2)
var<uniform> ubo : vec4<u32>; // x = numTris, y = maxDepth

// Helper: 8^x
fn pow8(exp: u32) -> u32 {
    // 8^x == 2^(3*x)
    return 1u << (3u * exp);
}

fn writeNode(i: u32, mn: vec3<f32>, mx: vec3<f32>) {
    let base = 1u + i * 6u;
    bvh[base + 0u] = mn.x;
    bvh[base + 1u] = mn.y;
    bvh[base + 2u] = mn.z;
    bvh[base + 3u] = mx.x;
    bvh[base + 4u] = mx.y;
    bvh[base + 5u] = mx.z;
}

fn getTriangleBounds(ti: u32) -> array<vec3<f32>, 2u> {
    let base = ti * 9u;
    var mn = vec3<f32>( 1e30);
    var mx = vec3<f32>(-1e30);

    for (var k = 0u; k < 3u; k = k + 1u) {
        let v = vec3<f32>(
            triangles[base + k * 3u + 0u],
            triangles[base + k * 3u + 1u],
            triangles[base + k * 3u + 2u]
        );
        mn = min(mn, v);
        mx = max(mx, v);
    }
    return array<vec3<f32>, 2u>(mn, mx);
}

// Get depth of node in an 8-ary heap
// Root(0) = Depth 0
// 1..8 = Depth 1
// 9..72 = Depth 2
fn nodeDepth(node: u32) -> u32 {
    var d = 0u;
    // Geometric sum inverse is messy, simpler to loop for low depths
    // Start range for depth k is (8^k - 1) / 7
    
    // D0: 0
    // D1: 1
    // D2: 9
    // D3: 73
    var limit = 0u;
    loop {
        // limit for *next* depth
        let countAtDepth = pow8(d);
        limit = limit + countAtDepth; // This limit is actually start of d+1
        if (node < limit) {
            return d;
        }
        d = d + 1u;
        if (d > 10u) { break; } // safety
    }
    return d;
}

fn getNodeTriRange(node: u32, numTris: u32, maxDepth: u32) -> vec2<u32> {
    let depth = nodeDepth(node);
    if (depth > maxDepth) { return vec2<u32>(0u, 0u); }

    // Range of nodes at this specific depth
    let startNodeAtDepth = (pow8(depth) - 1u) / 7u;
    let nodeOffset = node - startNodeAtDepth;

    // Total leaves at maxDepth
    let totalLeaves = pow8(maxDepth);
    
    // Leaves covered by one node at current depth
    // = 8^(maxDepth - depth)
    let leavesPerNode = pow8(maxDepth - depth);

    // Implicit mapping: Leaf Index L maps to Triangle Index L.
    // (Since leaves are just sequential in this layout)
    var startTri = nodeOffset * leavesPerNode;
    var endTri = startTri + leavesPerNode;

    if (startTri > numTris) { startTri = numTris; }
    if (endTri > numTris) { endTri = numTris; }

    return vec2<u32>(startTri, endTri);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let numTris  = ubo.x;
    let maxDepth = ubo.y; 

    // Total nodes in full 8-ary tree of depth maxDepth
    // Sum 8^i for i=0..maxDepth = (8^(maxDepth+1) - 1) / 7
    let totalNodes = (pow8(maxDepth + 1u) - 1u) / 7u;

    if (gid.x == 0u) {
        bvh[0] = f32(totalNodes);
    }

    let node = gid.x;
    if (node >= totalNodes) { return; }

    // Get triangle range
    let triRange = getNodeTriRange(node, numTris, maxDepth);
    let startTri = triRange.x;
    let endTri   = triRange.y;

    if (startTri >= endTri) {
        // Empty node
        writeNode(node, vec3<f32>(1e30), vec3<f32>(-1e30));
        return;
    }

    var mn = vec3<f32>( 1e30);
    var mx = vec3<f32>(-1e30);

    for(var ti = startTri; ti < endTri; ti = ti + 1u) {
        let ab = getTriangleBounds(ti);
        mn = min(mn, ab[0u]);
        mx = max(mx, ab[1u]);
    }

    writeNode(node, mn, mx);
}
// BVH4 Bottom-Up Builder (explicit children)
// Header:
//   bvh[0] = totalNodes
//   bvh[1] = rootIndex
// Node i:
//   base = 2u + i * 4u
//   bvh[base+0..2] = packed FP16 AABB
//   bvh[base+3]    = meta: (triIndex<<4) | (isLeaf<<3) | count(0..7)
//
// children buffer:
//   children[i*4 + k] = child index or 0xFFFFFFFFu

@group(0) @binding(0) var<storage, read_write> bvh : array<u32>;
@group(0) @binding(1) var<storage, read> triangles : array<f32>;
@group(0) @binding(2) var<uniform> ubo : vec4<u32>;
// ubo.x = numTris
// ubo.y = baseNode (where to write new nodes, for parent pass)
// ubo.z = groupsOffsetU32 (offset into groups buffer, in u32 units)
// ubo.w = groupsCount (number of parents to build this pass; 0 for leaf pass)

@group(0) @binding(3) var<storage, read_write> children : array<u32>;
@group(0) @binding(4) var<storage, read> groups : array<u32>; // groupsCount * 4 entries

const INVALID : u32 = 0xFFFFFFFFu;

// -----------------------------------------
// Helpers
// -----------------------------------------

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

fn decodeBounds(i: u32) -> array<vec3<f32>, 2u> {
    let base = 2u + (i << 2u);

    let a = unpack2x16float(bvh[base + 0u]); // mn.x, mn.y
    let b = unpack2x16float(bvh[base + 1u]); // mn.z, mx.x
    let c = unpack2x16float(bvh[base + 2u]); // mx.y, mx.z

    let mn = vec3<f32>(a.x, a.y, b.x);
    let mx = vec3<f32>(b.y, c.x, c.y);
    return array<vec3<f32>, 2u>(mn, mx);
}

fn writePackedBounds(i: u32, mn: vec3<f32>, mx: vec3<f32>) {
    let base = 2u + (i << 2u);

    // You can tune eps if you want conservative boxes
    let eps = 0.0f;
    let mnL = mn - eps;
    let mxL = mx + eps;

    bvh[base + 0u] = pack2x16float(vec2(mnL.x, mnL.y));
    bvh[base + 1u] = pack2x16float(vec2(mnL.z, mxL.x));
    bvh[base + 2u] = pack2x16float(vec2(mxL.y, mxL.z));
}

fn writeMeta(i: u32, isLeaf: u32, count: u32, triIndex: u32) {
    let base = 2u + (i << 2u);

    // meta: (triIndex << 4) | (isLeaf << 3) | (count & 7)
    let metadata = (triIndex << 4u) | ((isLeaf & 1u) << 3u) | (count & 7u);
    bvh[base + 3u] = metadata;
}

fn writeChildren(i: u32, c0: u32, c1: u32, c2: u32, c3: u32) {
    let o = i << 2u;
    children[o + 0u] = c0;
    children[o + 1u] = c1;
    children[o + 2u] = c2;
    children[o + 3u] = c3;
}

// -----------------------------------------
// Pass 1: Leaves (1 triangle = 1 leaf)
// -----------------------------------------
@compute @workgroup_size(256)
fn buildLeaves(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ti = gid.x;
    let numTris = ubo.x;

    if (ti >= numTris) { return; }

    let ab = getTriangleBounds(ti);
    let mn = ab[0u];
    let mx = ab[1u];

    writePackedBounds(ti, mn, mx);
    writeMeta(ti, 1u, 1u, ti); // leaf, triCount=1, triIndex=ti
    writeChildren(ti, INVALID, INVALID, INVALID, INVALID);
}

// -----------------------------------------
// Pass K: Parents from groups[] (bottom-up)
// groups: for each parent p: 4 child indices (INVALID allowed)
// parentIndex = baseNode + p
// -----------------------------------------
@compute @workgroup_size(256)
fn buildParents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    let baseNode = ubo.y;
    let groupsOffset = ubo.z;
    let groupsCount = ubo.w;

    if (p >= groupsCount) { return; }

    let parentIndex = baseNode + p;
    let gbase = groupsOffset + p * 4u;

    let c0 = groups[gbase + 0u];
    let c1 = groups[gbase + 1u];
    let c2 = groups[gbase + 2u];
    let c3 = groups[gbase + 3u];

    var mn = vec3<f32>( 1e30);
    var mx = vec3<f32>(-1e30);
    var childCount: u32 = 0u;

    if (c0 != INVALID) {
        let bb = decodeBounds(c0);
        mn = min(mn, bb[0u]);
        mx = max(mx, bb[1u]);
        childCount = childCount + 1u;
    }
    if (c1 != INVALID) {
        let bb = decodeBounds(c1);
        mn = min(mn, bb[0u]);
        mx = max(mx, bb[1u]);
        childCount = childCount + 1u;
    }
    if (c2 != INVALID) {
        let bb = decodeBounds(c2);
        mn = min(mn, bb[0u]);
        mx = max(mx, bb[1u]);
        childCount = childCount + 1u;
    }
    if (c3 != INVALID) {
        let bb = decodeBounds(c3);
        mn = min(mn, bb[0u]);
        mx = max(mx, bb[1u]);
        childCount = childCount + 1u;
    }

    // Should never happen (CPU should avoid single-child parents),
    // but keep it safe
    if (childCount == 0u) {
        writePackedBounds(parentIndex, vec3<f32>(1e30), vec3<f32>(-1e30));
        writeMeta(parentIndex, 0u, 0u, 0u);
        writeChildren(parentIndex, INVALID, INVALID, INVALID, INVALID);
        return;
    }

    writePackedBounds(parentIndex, mn, mx);
    writeMeta(parentIndex, 0u, childCount, 0u); // internal
    writeChildren(parentIndex, c0, c1, c2, c3);
}
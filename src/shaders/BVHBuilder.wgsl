// ─────────────────────────────────────────────
// Parallel BVH Builder (Bottom-Up)
// One thread per node
// Leaves store triangle AABBs
// Internal nodes merge child AABBs
// ─────────────────────────────────────────────

@group(0) @binding(0)
var<storage, read_write> bvh : array<f32>;    // [nodeMin.xyz, nodeMax.xyz] per node

@group(0) @binding(1)
var<storage, read> triangles : array<f32>;    // triangle float array

@group(0) @binding(2)
var<uniform> ubo : vec4<u32>; // x=numTris, y=maxDepth, z=batch, w=unused


// -----------------------------------------
// Helper functions
// -----------------------------------------

fn writeNode(i: u32, mn: vec3<f32>, mx: vec3<f32>) {
    let base = 1u + i*6u;
    bvh[base+0u] = mn.x;
    bvh[base+1u] = mn.y;
    bvh[base+2u] = mn.z;
    bvh[base+3u] = mx.x;
    bvh[base+4u] = mx.y;
    bvh[base+5u] = mx.z;
}

fn readNode(i: u32) -> array<vec3<f32>,2> {
    let base = 1u + i*6u;
    return array<vec3<f32>,2>(
        vec3<f32>(bvh[base+0u], bvh[base+1u], bvh[base+2u]),
        vec3<f32>(bvh[base+3u], bvh[base+4u], bvh[base+5u])
    );
}

fn getTriangleBounds(ti: u32) -> array<vec3<f32>,2> {
    let base = ti*9u;

    var mn = vec3<f32>( 1e30);
    var mx = vec3<f32>(-1e30);

    for (var k = 0u; k < 3u; k++) {
        let v = vec3<f32>(
            triangles[base + k*3u + 0u],
            triangles[base + k*3u + 1u],
            triangles[base + k*3u + 2u]
        );
        mn = min(mn, v);
        mx = max(mx, v);
    }

    return array<vec3<f32>,2>(mn, mx);
}


// -----------------------------------------
// MAIN ENTRY – fully parallel
// -----------------------------------------

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {

    let numTris = ubo.x;
    let maxDepth = ubo.y;

    let totalNodes = (1u << (maxDepth+1u)) - 1u;
    let leafStart  = (1u << maxDepth) - 1u;
    let numLeaves  = 1u << maxDepth;

    if (gid.x == 0u) {
        // store metadata count
        bvh[0] = f32(totalNodes);
    }

    if (gid.x >= totalNodes) {
        return;
    }

    // -----------------------------------------
    // 1) LEAF NODES (parallel)
    // -----------------------------------------
    if (gid.x >= leafStart) {
        let leafIdx = gid.x - leafStart;

        if (leafIdx < numTris) {
            let ab = getTriangleBounds(leafIdx);
            writeNode(gid.x, ab[0], ab[1]);
        } else {
            // Empty leaf bbox
            writeNode(gid.x,
                vec3<f32>( 1e30),
                vec3<f32>(-1e30)
            );
        }
        return;
    }

    // -----------------------------------------
    // 2) INTERNAL NODES (parallel)
    // -----------------------------------------

    // children indices
    let left  = gid.x*2u + 1u;
    let right = gid.x*2u + 2u;

    // merge AABBs
    let L = readNode(left);
    let R = readNode(right);

    let mn = min(L[0], R[0]);
    let mx = max(L[1], R[1]);

    writeNode(gid.x, mn, mx);
}
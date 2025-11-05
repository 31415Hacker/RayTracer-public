// ─────────────────────────────
// BVH encoded as a flat float array
// Each node = 6 floats: min.xyz, max.xyz
// ─────────────────────────────
@group(0) @binding(0)
var<storage, read_write> bvh : array<f32>;

@group(0) @binding(1)
var<storage, read> triangles : array<f32>;

// ─────────────────────────────
// Uniform buffer for global build parameters
// ─────────────────────────────
struct BuilderUBO {
    numTriangles : u32,
    maxDepth     : u32,
    batchSize    : u32,   // ← new: how many nodes per thread
    _pad         : u32,
};
@group(0) @binding(2)
var<uniform> ubo : BuilderUBO;

// ─────────────────────────────
// Helpers
// ─────────────────────────────
fn getTriangle(i: u32) -> array<vec3<f32>, 3> {
    let base = i * 9u;
    return array<vec3<f32>, 3>(
        vec3<f32>(triangles[base + 0u], triangles[base + 1u], triangles[base + 2u]),
        vec3<f32>(triangles[base + 3u], triangles[base + 4u], triangles[base + 5u]),
        vec3<f32>(triangles[base + 6u], triangles[base + 7u], triangles[base + 8u])
    );
}

fn writeNode(nodeIndex: u32, mn: vec3<f32>, mx: vec3<f32>) {
    let base = nodeIndex * 6u;
    bvh[base + 0u] = mn.x;
    bvh[base + 1u] = mn.y;
    bvh[base + 2u] = mn.z;
    bvh[base + 3u] = mx.x;
    bvh[base + 4u] = mx.y;
    bvh[base + 5u] = mx.z;
}

fn readNode(nodeIndex: u32) -> array<vec3<f32>, 2> {
    let base = nodeIndex * 6u;
    return array<vec3<f32>, 2>(
        vec3<f32>(bvh[base + 0u], bvh[base + 1u], bvh[base + 2u]),
        vec3<f32>(bvh[base + 3u], bvh[base + 4u], bvh[base + 5u])
    );
}

fn leftChild(i: u32) -> u32 { return 2u * i + 1u; }
fn rightChild(i: u32) -> u32 { return 2u * i + 2u; }

// ─────────────────────────────
// Compute Entry
// ─────────────────────────────
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batchSize = ubo.batchSize;
    let threadBase = gid.x * batchSize;

    // Only thread 0 builds the root (once)
    if (gid.x == 0u) {
        var globalMin = vec3<f32>( 1e20);
        var globalMax = vec3<f32>(-1e20);

        for (var i = 0u; i < ubo.numTriangles; i = i + 1u) {
            let tri = getTriangle(i);
            for (var v = 0u; v < 3u; v = v + 1u) {
                globalMin = min(globalMin, tri[v]);
                globalMax = max(globalMax, tri[v]);
            }
        }
        writeNode(0u, globalMin, globalMax);
    }

    workgroupBarrier();

    // ─────────────────────────────
    // Breadth-first subdivision (batched)
    // ─────────────────────────────
    let maxDepth = ubo.maxDepth;

    var currentLevelStart: u32 = 0u;
    var currentLevelEnd:   u32 = 1u; // root only

    for (var level = 0u; level < maxDepth; level = level + 1u) {
        let nextLevelStart = currentLevelEnd;
        let nextLevelEnd   = nextLevelStart + (currentLevelEnd - currentLevelStart) * 2u;
        let numNodes       = currentLevelEnd - currentLevelStart;

        // Process nodes in batches
        for (var j = 0u; j < batchSize; j = j + 1u) {
            let nodeIndex = currentLevelStart + threadBase + j;
            if (nodeIndex >= currentLevelEnd) { break; }

            let node = readNode(nodeIndex);
            let pMin = node[0];
            let pMax = node[1];
            let extent = pMax - pMin;

            // Split along longest axis
            var axis: u32 = 0u;
            if (extent.y > extent.x && extent.y > extent.z) {
                axis = 1u;
            } else if (extent.z > extent.x) {
                axis = 2u;
            }

            let mid = pMin[axis] + 0.5 * extent[axis];

            // Left child
            var lMin = pMin;
            var lMax = pMax;
            lMax[axis] = mid;

            // Right child
            var rMin = pMin;
            var rMax = pMax;
            rMin[axis] = mid;

            writeNode(leftChild(nodeIndex),  lMin, lMax);
            writeNode(rightChild(nodeIndex), rMin, rMax);
        }

        workgroupBarrier();

        currentLevelStart = nextLevelStart;
        currentLevelEnd   = nextLevelEnd;
    }
}
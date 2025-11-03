// Define the data structure for BVH nodes
struct BVHNodes {
    nodes: array<vec4<f32>>,
};

// Bind the storage buffer
@group(0) @binding(0)
var<storage, read_write> bvhNodes : BVHNodes;

// Compute shader entry point
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;

    bvhNodes.nodes[index] = vec4<f32>(f32(index), 0.0, 0.0, 1.0);
}
// ─────────────────────────────
// Data structures
// ─────────────────────────────
struct UBO {
    resolution: vec4<f32>,      // xy = width + height, z = focal, w = padding
    camPosNumTris: vec4<f32>,   // xyz = position, w = numTris
    camQuat: vec4<f32>,         // xyzw = quaternion
    nodes: vec4<f32>,           // x = totalNodes, y = maxDepth, zw = padding
};

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    invdirection: vec3<f32>,
};

struct Triangle {
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
};

struct BVHNode {
    min: vec3<f32>,
    max: vec3<f32>,
};

// ─────────────────────────────
// Bindings
// ─────────────────────────────
@group(0) @binding(0)
var outputTexture : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1)
var<uniform> ubo : UBO;
@group(0) @binding(2)
var<storage, read> triangles : array<f32>;
@group(0) @binding(3)
var<storage, read> BVH : array<f32>; 

// ─────────────────────────────
// Math helpers
// ─────────────────────────────
fn rotateVectorByQuat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u +
           (s * s - dot(u, u)) * v +
           2.0 * s * cross(u, v);
}

fn shade(normal: vec3<f32>, lightDir: vec3<f32>) -> vec3<f32> {
    let ndotl = max(dot(normal, lightDir), 0.0);
    return vec3<f32>(ndotl + 0.1);
}

// ─────────────────────────────
// Geometry accessors
// ─────────────────────────────
fn getTriangle(index: u32) -> Triangle {
    let base = index * 9u;
    return Triangle(
        vec3<f32>(triangles[base], triangles[base + 1u], triangles[base + 2u]),
        vec3<f32>(triangles[base + 3u], triangles[base + 4u], triangles[base + 5u]),
        vec3<f32>(triangles[base + 6u], triangles[base + 7u], triangles[base + 8u])
    );
}

fn getBVHNode(index: u32) -> BVHNode {
    let base = 1u + index * 6u;
    return BVHNode(
        vec3<f32>(BVH[base], BVH[base + 1u], BVH[base + 2u]),
        vec3<f32>(BVH[base + 3u], BVH[base + 4u], BVH[base + 5u])
    );
}

// ─────────────────────────────
// Intersections
// ─────────────────────────────

// Returns distance to AABB. 
// Returns 1e30 if miss OR if distance is > curMinT (optimization).
fn intersectAABB(ray: Ray, minB: vec3<f32>, maxB: vec3<f32>, curMinT: f32) -> f32 {
    let t1 = (minB - ray.origin) * ray.invdirection;
    let t2 = (maxB - ray.origin) * ray.invdirection;

    // We use min/max which are essentially hardware 'select' calls
    let tminVec = min(t1, t2);
    let tmaxVec = max(t1, t2);

    let tmin = max(max(tminVec.x, tminVec.y), tminVec.z);
    let tmax = min(min(tmaxVec.x, tmaxVec.y), tmaxVec.z);

    // Logic: 
    // 1. tmax < 0.0: Box is behind ray
    // 2. tmin > tmax: Ray misses box
    // 3. tmin > curMinT: Box is further than closest triangle hit
    
    // Valid hit condition:
    let hit = (tmax >= 0.0) && (tmax >= tmin) && (tmin <= curMinT);

    // Use select(falseVal, trueVal, condition)
    // If hit is true, return max(tmin, 0.0), else return 1e30
    return select(1e30, max(tmin, 0.0), hit);
}

fn intersectRayTriangle(rayOrigin: vec3<f32>, rayDir: vec3<f32>, tri: Triangle) -> f32 {
    let eps = 1e-10;
    let edge1 = tri.v1 - tri.v0;
    let edge2 = tri.v2 - tri.v0;
    let h = cross(rayDir, edge2);
    let a = dot(edge1, h);
    
    if (abs(a) < eps) { return -1.0; }

    let f = 1.0 / a;
    let s = rayOrigin - tri.v0;
    let u = f * dot(s, h);
    
    if (u < 0.0 || u > 1.0) { return -1.0; }

    let q = cross(s, edge1);
    let v = f * dot(rayDir, q);
    
    if (v < 0.0 || (u + v) > 1.0) { return -1.0; }

    let t = f * dot(edge2, q);
    
    // Branchless return using select? 
    // WGSL doesn't support early return inside select, so we keep the if logic for flow control efficiency
    if (t > eps) { return t; }

    return -1.0;
}

// ─────────────────────────────
// BVH8 Traversal (Oct-Tree)
// ─────────────────────────────
fn traverseBVH(ray: Ray, numTris: u32) -> i32 {
    let numNodes = u32(BVH[0]);
    
    // If empty BVH, return -1
    if (numNodes == 0u) { return -1; }

    let maxDepth = u32(ubo.nodes.y);
    
    // Calculate start of Leaf level
    // Series: 8^0 + 8^1 + ... + 8^(maxDepth-1) nodes are internal
    // leafStart = (8^maxDepth - 1) / 7
    let leafStart = ((1u << (3u * maxDepth)) - 1u) / 7u;

    // STACK FIX: 64 is safe for depth ~6 Oct-Tree
    var stack: array<u32, 16>; 
    var stackPtr: i32 = 0;
    stack[0] = 0u; // Push Root

    var closestTri = -1;
    var minT = 1e20;

    // Initial cull: Check root box
    let rootNode = getBVHNode(0u);
    if (intersectAABB(ray, rootNode.min, rootNode.max, minT) >= 1e30) {
        return -1;
    }

    loop {
        if (stackPtr < 0) { break; }
        
        let nodeIndex = stack[stackPtr];
        stackPtr = stackPtr - 1;

        let isLeaf = (nodeIndex >= leafStart);

        if (isLeaf) {
            // Leaf node in Implicit Oct-Tree maps directly to Triangle Index
            let triIndex = nodeIndex - leafStart;
            
            // Bounds check against total triangles
            if (triIndex < numTris) {
                let node = getBVHNode(nodeIndex);
                
                // Re-check AABB with current minT to skip unnecessary triangle math
                if (intersectAABB(ray, node.min, node.max, minT) < 1e30) {
                    let tri = getTriangle(triIndex);
                    let t = intersectRayTriangle(ray.origin, ray.direction, tri);
                    
                    // Update closest hit
                    if (t > 0.0 && t < minT) {
                        minT = t;
                        closestTri = i32(triIndex);
                    }
                }
            }
        } else {
            // Internal Node: Process 8 children
            // Children indices: nodeIndex * 8 + 1 ... + 8
            let firstChild = nodeIndex * 8u + 1u;

            // Iterate 7 down to 0 so we push in reverse order (stack LIFO)
            for (var i = 7u; i < 8u; i = i - 1u) { 
                let childIdx = firstChild + i;
                
                if (childIdx < numNodes) {
                    let childNode = getBVHNode(childIdx);
                    let dist = intersectAABB(ray, childNode.min, childNode.max, minT);
                    
                    if (dist < 1e30) {
                        stackPtr = stackPtr + 1;
                        // Safety clamp
                        if (stackPtr < 16) {
                            stack[stackPtr] = childIdx;
                        }
                    }
                }
                
                // Manual break for u32 loop wrapping
                if (i == 0u) { break; }
            }
        }
    }

    return closestTri;
}

// ─────────────────────────────
// Main compute entry
// ─────────────────────────────
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixelCoord = vec2<i32>(gid.xy);
    let resolution = ubo.resolution.xy;

    if (pixelCoord.x >= i32(resolution.x) || pixelCoord.y >= i32(resolution.y)) {
        return;
    }

    // Camera Setup
    let aspect = resolution.x / resolution.y;
    // let fov = radians(70.0);
    let focal = 1.428148;// 1.0 / tan(fov * 0.5);

    let uv = vec2<f32>(
        f32(pixelCoord.x) / resolution.x,
        f32(pixelCoord.y) / resolution.y
    );
    let px = uv.x * 2.0 - 1.0;
    let py = uv.y * 2.0 - 1.0;

    // Ray Direction
    let dirCamera = normalize(vec3<f32>(px * aspect, py, -focal));
    let camQuat = ubo.camQuat;
    let dir = rotateVectorByQuat(dirCamera, camQuat);
    let camPos = ubo.camPosNumTris.xyz;
    
    // Safe inverse direction (avoid div by zero)
    let invDir = 1.0 / (dir + vec3<f32>(select(0.0, 1e-5, dir.x == 0.0), select(0.0, 1e-5, dir.y == 0.0), select(0.0, 1e-5, dir.z == 0.0))); 
    
    let ray = Ray(camPos, dir, invDir);

    // Lighting Setup
    let lightDir = normalize(vec3<f32>(-0.8, 0.5, 1.0));
    let numTris = u32(ubo.camPosNumTris.w);

    // Traversal
    let hitTri = traverseBVH(ray, numTris);

    // Shading
    // We cannot use select() safely for memory access (getTriangle) 
    // because select evaluates both branches. We must use 'if'.
    var color = vec3<f32>(0.01, 0.01, 0.01); // Background color

    if (hitTri >= 0) {
        let tri = getTriangle(u32(hitTri));
        let n = normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
        color = shade(n, lightDir);
    }

    textureStore(outputTexture, pixelCoord, vec4<f32>(color, 1.0));
}
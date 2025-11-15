// ─────────────────────────────
// Data structures
// ─────────────────────────────
struct UBO {
    resolution: vec4<f32>,
    camPosNumTris: vec4<f32>,   // xyz = position, w = numTris
    camQuat: vec4<f32>,         // xyzw = quaternion
    nodes: vec4<f32>,          // x = numNodes, yzw = padding
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
fn cross(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

fn rotateVectorByQuat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u +
           (s * s - dot(u, u)) * v +
           2.0 * s * cross(u, v);
}

fn shade(normal: vec3<f32>, lightDir: vec3<f32>) -> vec3<f32> {
    let ndotl = max(dot(normal, lightDir), 0.0);
    return vec3<f32>(ndotl);
}

// ─────────────────────────────
// Geometry accessors
// ─────────────────────────────
fn getTriangle(index: u32) -> Triangle {
    let base = index * 9u;
    return Triangle(
        vec3<f32>(triangles[base + 0u], triangles[base + 1u], triangles[base + 2u]),
        vec3<f32>(triangles[base + 3u], triangles[base + 4u], triangles[base + 5u]),
        vec3<f32>(triangles[base + 6u], triangles[base + 7u], triangles[base + 8u])
    );
}

fn getBVHNode(index: u32) -> BVHNode {
    let base = index * 6u;
    return BVHNode(
        vec3<f32>(BVH[base + 0u], BVH[base + 1u], BVH[base + 2u]),
        vec3<f32>(BVH[base + 3u], BVH[base + 4u], BVH[base + 5u])
    );
}

// ─────────────────────────────
// Ray–AABB intersection
// ─────────────────────────────
fn intersectAABB(ray: Ray, minB: vec3<f32>, maxB: vec3<f32>) -> bool {
    let t1 = (minB - ray.origin) * ray.invdirection;
    let t2 = (maxB - ray.origin) * ray.invdirection;

    let tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));

    return tmax >= max(tmin, 0.0);
}

// ─────────────────────────────
// Ray–Triangle intersection (Möller–Trumbore)
// ─────────────────────────────
fn intersectRayTriangle(rayOrigin: vec3<f32>, rayDir: vec3<f32>, tri: Triangle) -> f32 {
    let eps = 1e-6;
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
    if (t > eps) { return t; }

    return -1.0;
}

// ─────────────────────────────
// BVH traversal
// ─────────────────────────────
fn traverseBVH(ray: Ray, numNodes: u32, numTris: u32) -> i32 {
    var stack: array<u32, 64>;
    var stackPtr: i32 = 0;
    stack[0] = 0u;

    var closestTri = -1;
    var minT = 1e20;

    loop {
        if (stackPtr < 0) { break; }

        let nodeIndex = stack[stackPtr];
        stackPtr -= 1;

        let node = getBVHNode(nodeIndex);
        if (!intersectAABB(ray, node.min, node.max)) {
            if (stackPtr < 0) { break; }
            continue;
        }

        // compute child indices
        let left  = nodeIndex * 2u + 1u;
        let right = nodeIndex * 2u + 2u;

        // leaf detection: both children out of range
        let isLeaf = (left >= numNodes) && (right >= numNodes);

        if (isLeaf) {
            // Brute-force over triangles (OK while numTris is small)
            for (var i: u32 = 0u; i < numTris; i = i + 1u) {
                let tri = getTriangle(i);
                let c = (tri.v0 + tri.v1 + tri.v2) / 3.0;

                if (all(c >= node.min) && all(c <= node.max)) {
                    let t = intersectRayTriangle(ray.origin, ray.direction, tri);
                    if (t > 0.0 && t < minT) {
                        minT = t;
                        closestTri = i32(i);
                    }
                }
            }
        } else {
            // push valid children
            if (right < numNodes) {
                stackPtr += 1;
                stack[stackPtr] = right;
            }
            if (left < numNodes) {
                stackPtr += 1;
                stack[stackPtr] = left;
            }
        }

        if (stackPtr < 0) { break; }
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

    // Guard: pixel outside viewport
    if (pixelCoord.x >= i32(resolution.x) || pixelCoord.y >= i32(resolution.y)) {
        return;
    }

    // Screen / camera setup
    let aspect = resolution.x / resolution.y;
    let fov = radians(70.0);
    let focal = 1.0 / tan(fov * 0.5);

    // Normalized pixel coord [0..1]
    let uv = vec2<f32>(
        f32(pixelCoord.x) / resolution.x,
        f32(pixelCoord.y) / resolution.y
    );

    // Convert to NDC [-1..1]
    let px = uv.x * 2.0 - 1.0;
    let py = uv.y * 2.0 - 1.0;   // Y-flip for screen space

    // Camera-space ray direction (THREE.js perspective)
    let dirCamera = normalize(vec3<f32>(
        px * aspect,
        py,
        -focal                    // THREE.js looks down -Z
    ));

    // Rotate by camera quaternion
    let camQuat = ubo.camQuat;
    var dir = rotateVectorByQuat(dirCamera, camQuat);

    // Build ray
    let camPos = ubo.camPosNumTris.xyz;
    let ray = Ray(camPos, dir, 1.0 / dir);

    // Lighting
    let lightDir = normalize(vec3<f32>(-0.8, 0.5, 1.0));

    // BVH info
    let numTris = u32(ubo.camPosNumTris.w);
    let numNodes = u32(ubo.nodes.x);

    // BVH traversal
    let hitTri = traverseBVH(ray, numNodes, numTris);

    // Shading
    var color: vec3<f32>;
    if (hitTri >= 0) {
        let tri = getTriangle(u32(hitTri));
        let n = normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
        let diffuse = shade(n, lightDir);
        color = diffuse;
    } else {
        color = vec3<f32>(0.0, 0.0, 0.0);
    }

    textureStore(outputTexture, pixelCoord, vec4<f32>(color, 1.0));
}
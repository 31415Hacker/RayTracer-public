// ─────────────────────────────
// Data structures
// ─────────────────────────────
struct BVHNodes {
    nodes: array<vec4<f32>>,
};

struct UBO {
    resolution: vec4<f32>,
    camPos: vec4<f32>,       // xyz = position, w = unused
    camQuat: vec4<f32>,      // xyzw = quaternion rotation
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

// ─────────────────────────────
// Bindings
// ─────────────────────────────
@group(0) @binding(0)
var outputTexture : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1)
var<uniform> ubo : UBO;

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
    // q = (x, y, z, w)
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u
         + (s * s - dot(u, u)) * v
         + 2.0 * s * cross(u, v);
}

fn shade(normal: vec3<f32>, lightDir: vec3<f32>) -> vec3<f32> {
    let ndotl = max(dot(normal, lightDir), 0.0);
    return vec3<f32>(ndotl);
}

// ─────────────────────────────
// Ray–Triangle Intersection (Möller–Trumbore)
// ─────────────────────────────
fn intersectRayTriangle(rayOrigin: vec3<f32>, rayDir: vec3<f32>, triangle: Triangle) -> f32 {
    let eps = 1e-6;
    let edge1 = triangle.v1 - triangle.v0;
    let edge2 = triangle.v2 - triangle.v0;

    let h = cross(rayDir, edge2);
    let a = dot(edge1, h);
    if (abs(a) < eps) {
        return -1.0;
    }

    let f = 1.0 / a;
    let s = rayOrigin - triangle.v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }

    let q = cross(s, edge1);
    let v = f * dot(rayDir, q);
    if (v < 0.0 || (u + v) > 1.0) {
        return -1.0;
    }

    let t = f * dot(edge2, q);
    if (t > eps) {
        return t;
    }
    return -1.0;
}

// ─────────────────────────────
// Main compute entry
// ─────────────────────────────
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixelCoord = vec2<i32>(gid.xy);
    let resolution = ubo.resolution.xy;
    let aspect = resolution.x / resolution.y;
    if (pixelCoord.x >= i32(resolution.x) || pixelCoord.y >= i32(resolution.y)) {
        return;
    }

    // Normalized pixel coordinate [0, 1]
    let uv = vec2<f32>(f32(pixelCoord.x), f32(pixelCoord.y)) / resolution;

    // NDC [-1, 1]
    let ndc = vec2<f32>(
        (uv.x * 2.0 - 1.0) * aspect,
        1.0 - uv.y * 2.0
    );

    // Camera setup
    let camPos = ubo.camPos.xyz;
    let camQuat = ubo.camQuat;
    let forward = rotateVectorByQuat(vec3<f32>(0.0, 0.0, -1.0), camQuat);
    let right = rotateVectorByQuat(vec3<f32>(1.0, 0.0, 0.0), camQuat);
    let up = rotateVectorByQuat(vec3<f32>(0.0, 1.0, 0.0), camQuat);

    let dir = normalize(forward + ndc.x * right + ndc.y * up);
    let ray = Ray(camPos, dir, 1.0 / dir);

    // Triangles facing camera
    let triangles = array<Triangle, 3>(
        Triangle(vec3<f32>(-1.2, -0.5, -4.0),
                vec3<f32>(-0.7,  0.5, -4.0),
                vec3<f32>(-0.2, -0.5, -4.0)),

        Triangle(vec3<f32>(-0.3, -0.5, -4.0),
                vec3<f32>(0.2,   0.5, -4.0),
                vec3<f32>(0.7,  -0.5, -4.0)),

        Triangle(vec3<f32>(0.6,  -0.5, -4.0),
                vec3<f32>(1.1,   0.5, -4.0),
                vec3<f32>(1.6,  -0.5, -4.0))
    );
    
    let lightDir = normalize(vec3<f32>(0.8, 0.5, -1.0)); // toward -Z (camera side)

    var t: f32 = -1.0;
    var prevT: f32 = 1e20;
    var trisIndex: i32 = -1;

    // Intersection
    for (var i = 0; i < 3; i++) {
        t = intersectRayTriangle(ray.origin, ray.direction, triangles[i]);
        if (t < prevT && t > 0.00001) {
            trisIndex = i;
            prevT = t;
        }
    }

    var color: vec3<f32>;
    if (trisIndex >= 0) {
        let hitPos = ray.origin + ray.direction * prevT;
        let normal = normalize(cross(
            triangles[trisIndex].v1 - triangles[trisIndex].v0,
            triangles[trisIndex].v2 - triangles[trisIndex].v0
        ));
        let diffuse = shade(normal, lightDir);
        color = vec3<f32>(1.0, 1.0, 1.0) * diffuse;
    } else {
        let bg = 0.5 * (dir.y + 1.0);
        color = mix(vec3<f32>(0.2, 0.3, 0.5), vec3<f32>(0.8, 0.9, 1.0), bg);
    }

    textureStore(outputTexture, pixelCoord, vec4<f32>(color, 1.0));
}
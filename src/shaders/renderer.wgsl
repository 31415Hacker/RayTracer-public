// ============================================================
// Structs
// ============================================================

struct RendererUBO {
    resolution: vec4<f32>,
    // x: width, y: height, z: focal, w: aspect
    camPosNumTris: vec4<f32>,
    // xyz: camera position, w: num triangles
    camQuat: vec4<f32>,
    // camera orientation quaternion
};

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
    invdir: vec3<f32>,
};

struct Triangle {
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
};

struct BVHNode {
    min: vec3<f32>,
    max: vec3<f32>,
    firstTri: u32,
    triCount: u32,
};

struct Hit {
    t: f32,
    normal: vec3<f32>,
    hit: bool,
};

// ============================================================
// Bindings
// ============================================================

@group(0) @binding(0)
var outputTex: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var<uniform> ubo: RendererUBO;

@group(0) @binding(2)
var<storage, read> triangles: array<f32>;

@group(0) @binding(3)
var<storage, read> BVH: array<u32>;

// Debug counters (optional)
var<private> tris: u32 = 0u;
var<private> nodes: u32 = 0u;

// ============================================================
// Math Helpers
// ============================================================

fn rotateVectorByQuat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    let uv = cross(u, v);
    let uuv = cross(u, uv);
    return fma(vec3<f32>(2.0), fma(vec3<f32>(s), uv, uuv), v);
}

// ============================================================
// Data Access
// ============================================================

fn getTriangle(index: u32) -> Triangle {
    let b = index * 9u;
    return Triangle(vec3(triangles[b + 0u], triangles[b + 1u], triangles[b + 2u]), vec3(triangles[b + 3u], triangles[b + 4u], triangles[b + 5u]), vec3(triangles[b + 6u], triangles[b + 7u], triangles[b + 8u]));
}

fn getBVHNode(index: u32) -> BVHNode {
    let base = 1u + index * 4u;

    let a = unpack2x16float(BVH[base + 0u]);
    let b = unpack2x16float(BVH[base + 1u]);
    let c = unpack2x16float(BVH[base + 2u]);

    let mn = vec3(a.xy, b.x);
    let mx = vec3(b.y, c.xy);

    let bits = BVH[base + 3u];
    let firstTri = bits >> 3u;
    let triCount = bits & 0x7u;

    return BVHNode(mn, mx, firstTri, triCount);
}

// ============================================================
// Intersection
// ============================================================

fn intersectAABB(ray: Ray, mn: vec3<f32>, mx: vec3<f32>, maxT: f32) -> f32 {
    nodes++;

    if (any(mn > mx)) {
        return 1e30;
    }

    let t1 = (mn - ray.origin) * ray.invdir;
    let t2 = (mx - ray.origin) * ray.invdir;

    let tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));

    if (tmax >= max(tmin, 0.0) && tmin <= maxT) {
        return max(tmin, 0.0);
    }
    return 1e30;
}

fn intersectTriangle(ray: Ray, tri: Triangle) -> f32 {
    tris++;

    let eps = 1e-7;
    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;

    let p = cross(ray.dir, e2);
    let det = dot(e1, p);

    if (abs(det) < eps) {
        return -1.0;
    }

    let invDet = 1.0 / det;
    let s = ray.origin - tri.v0;
    let u = invDet * dot(s, p);

    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }

    let q = cross(s, e1);
    let v = invDet * dot(ray.dir, q);

    if (v < 0.0 || u + v > 1.0) {
        return -1.0;
    }

    let t = invDet * dot(e2, q);
    return select(-1.0, t, t > eps);
}

// ============================================================
// BVH Traversal
// ============================================================

fn traverseBVH(ray: Ray, numTris: u32) -> Hit {
    let numNodes = BVH[0u];
    if (numNodes == 0u || numTris == 0u) {
        return Hit(0.0, vec3(0.0), false);
    }

    var stack: array<u32, 64>;
    var sp: i32 = 0;
    stack[0] = 0u;

    var closestT = 1e20;
    var normal = vec3<f32>(0.0);
    var hit = false;

    loop {
        if (sp < 0) {
            break;
        }

        let nodeIndex = stack[u32(sp)];
        sp--;

        let node = getBVHNode(nodeIndex);
        if (intersectAABB(ray, node.min, node.max, closestT) >= 1e30) {
            continue;
        }

        if (node.triCount > 0u) {
            var i = node.firstTri;
            let end = min(i + node.triCount, numTris);

            loop {
                if (i >= end) {
                    break;
                }

                let tri = getTriangle(i);
                let t = intersectTriangle(ray, tri);

                if (t > 0.0 && t < closestT) {
                    closestT = t;
                    normal = normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                    hit = true;
                }
                i++;
            }
        }
        else {
            let firstChild = nodeIndex * 8u + 1u;

            for (var c = 0u; c < 8u; c++) {
                let ci = firstChild + c;
                if (ci < numNodes) {
                    let child = getBVHNode(ci);
                    let d = intersectAABB(ray, child.min, child.max, closestT);
                    if (d < 1e30 && sp < 63) {
                        sp++;
                        stack[u32(sp)] = ci;
                    }
                }
            }
        }
    }

    return Hit(closestT, normal, hit);
}

// ============================================================
// Shading
// ============================================================

fn shade(hit: Hit) -> vec3<f32> {
    let lightDir = normalize(vec3(1.0, 1.5, 1.0));
    let baseColor = vec3(0.9, 0.7, 0.3);

    let ndotl = max(dot(hit.normal, lightDir), 0.0);
    let ambient = baseColor * 0.15;
    let diffuse = baseColor * ndotl;

    // Optional debug: node cost visualization
    let debug = clamp(0.0001 * f32(nodes), 0.0, 1.0);

    return mix(diffuse + ambient, vec3(debug), 0.0);
}

// ============================================================
// Main
// ============================================================

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<u32>(u32(ubo.resolution.x), u32(ubo.resolution.y));
    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(res);
    let p = fma(uv, vec2<f32>(2.0), vec2<f32>(-1.0));

    let aspect = ubo.resolution.w;
    let focal = ubo.resolution.z;

    var dir = normalize(vec3<f32>(p.x * aspect, p.y, - focal));
    dir = rotateVectorByQuat(dir, ubo.camQuat);

    let ray = Ray(ubo.camPosNumTris.xyz, dir, 1.0 / dir);

    let hit = traverseBVH(ray, u32(ubo.camPosNumTris.w));
    let color = select(vec3<f32>(0.01), shade(hit), hit.hit);

    textureStore(outputTex, vec2<i32>(gid.xy), vec4(color, 1.0));
}
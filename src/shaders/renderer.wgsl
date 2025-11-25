// ----------------------------------
// Structs
// ----------------------------------

struct RendererUBO {
    // x: width, y: height, z: focal, w: aspect
    resolution: vec4<f32>,
    // x,y,z: Camera Position, w: Number of Triangles
    camPosNumTris: vec4<f32>,
    // Camera Orientation Quaternion (x, y, z, w)
    camQuat: vec4<f32>,
    // x: Number of Nodes (optional), y: Max BVH Depth (optional)
    nodes: vec4<f32>,
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

// ----------------------------------
// Bindings
// ----------------------------------

@group(0) @binding(0)
var outputTex : texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var<uniform> ubo : RendererUBO;

// Triangles buffer: 9 floats per triangle (v0.xyz, v1.xyz, v2.xyz)
@group(0) @binding(2)
var<storage, read> triangles : array<f32>;

// BVH buffer: BVH[0] = total node count. Then 8 floats per node.
@group(0) @binding(3)
var<storage, read> BVH : array<f32>;

// Per-pixel debug counters (private, no atomics → “micro-setting B”)
var<private> tris  = 0u;
var<private> nodes = 0u;

// ----------------------------------
// Helpers
// ----------------------------------

// Rotates a vector by a unit quaternion (q.xyz = vector, q.w = scalar)
fn rotateVectorByQuat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let u: vec3<f32> = q.xyz;
    let s: f32 = q.w;
    let uv: vec3<f32> = cross(u, v);
    let uuv: vec3<f32> = cross(u, uv);
    return v + (2.0 * fma(vec3(s), uv, uuv));
}

// Retrieves triangle data from the flattened storage buffer
fn getTriangle(index: u32) -> Triangle {
    let base: u32 = index * 9u;
    let v0: vec3<f32> = vec3<f32>(
        triangles[base + 0u],
        triangles[base + 1u],
        triangles[base + 2u]
    );
    let v1: vec3<f32> = vec3<f32>(
        triangles[base + 3u],
        triangles[base + 4u],
        triangles[base + 5u]
    );
    let v2: vec3<f32> = vec3<f32>(
        triangles[base + 6u],
        triangles[base + 7u],
        triangles[base + 8u]
    );
    return Triangle(v0, v1, v2);
}

// Retrieves BVH node data from the flattened storage buffer
fn getBVHNode(index: u32) -> BVHNode {
    // Base index calculation: 1 (for count) + index * 8 (8 floats per node)
    let base = 1u + index * 8u;

    let mn = vec3<f32>(
        BVH[base + 0u],
        BVH[base + 1u],
        BVH[base + 2u]
    );
    let mx = vec3<f32>(
        BVH[base + 3u],
        BVH[base + 4u],
        BVH[base + 5u]
    );

    let firstTri = u32(BVH[base + 6u]);
    let triCount = u32(BVH[base + 7u]);

    return BVHNode(mn, mx, firstTri, triCount);
}

// AABB intersection (Slab method)
// Returns tmin of intersection, or 1e30 (miss) if no hit or hit is past curMinT
fn intersectAABB(
    ray: Ray,
    invDir: vec3<f32>,
    mn: vec3<f32>,
    mx: vec3<f32>,
    curMinT: f32
) -> f32 {
    nodes += 1u;

    // Quickly reject empty nodes written by BVHBuilder
    if (all(mn == vec3<f32>(1e30)) || all(mx == vec3<f32>(-1e30))) {
        return 1e30;
    }

    let t1: vec3<f32> = (mn - ray.origin) * invDir;
    let t2: vec3<f32> = (mx - ray.origin) * invDir;

    let tminVec: vec3<f32> = min(t1, t2);
    let tmaxVec: vec3<f32> = max(t1, t2);

    let tmin: f32 = max(max(tminVec.x, tminVec.y), tminVec.z);
    let tmax: f32 = min(min(tmaxVec.x, tmaxVec.y), tmaxVec.z);

    // Hit condition
    let hit: bool = (tmax >= tmin) && (tmax >= 0.0) && (tmin <= curMinT);

    if (hit) {
        return max(tmin, 0.0);
    }
    return 1e30;
}

// Möller-Trumbore ray-triangle intersection
// Returns t-value > 0.0 on hit, or -1.0 on miss
fn intersectRayTriangle(ray: Ray, tri: Triangle) -> f32 {
    tris += 1u;

    let eps: f32 = 1e-7;
    let edge1: vec3<f32> = tri.v1 - tri.v0;
    let edge2: vec3<f32> = tri.v2 - tri.v0;

    let h: vec3<f32> = cross(ray.dir, edge2);
    let a: f32 = dot(edge1, h);

    if (abs(a) < eps) {
        return -1.0;
    }

    let f: f32 = 1.0 / a;
    let s: vec3<f32> = ray.origin - tri.v0;
    let u: f32 = f * dot(s, h);

    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }

    let q: vec3<f32> = cross(s, edge1);
    let v: f32 = f * dot(ray.dir, q);

    if (v < 0.0 || u + v > 1.0) {
        return -1.0;
    }

    let t: f32 = f * dot(edge2, q);
    if (t <= eps) {
        return -1.0;
    }

    return t;
}

// Traverse the flattened 8-ary BVH structure using a stack
// Returns vec4<f32>(color.rgb, t_distance)
fn traverseBVH(ray: Ray, numTris: u32) -> vec4<f32> {
    const MAX_STACK_SIZE: u32 = 64u; // safe for maxDepth <= 7

    let numNodes = u32(BVH[0u]);
    if (numNodes == 0u || numTris == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }

    // Stack and pointer
    var stack: array<u32, MAX_STACK_SIZE>;
    var sp: i32 = 0;
    stack[0] = 0u; // root

    var closestT: f32 = 1e20;
    var hitNormal: vec3<f32> = vec3<f32>(0.0);
    var hitFound: bool = false;

    let invDir = ray.invdir;

    // Root culling
    let root = getBVHNode(0u);
    if (intersectAABB(ray, invDir, root.min, root.max, closestT) >= 1e30) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }

    // Traversal loop
    loop {
        if (sp < 0) {
            break;
        }

        let nodeIndex = stack[u32(sp)];
        sp = sp - 1;

        let node = getBVHNode(nodeIndex);

        // Cull this node
        if (intersectAABB(ray, invDir, node.min, node.max, closestT) >= 1e30) {
            continue;
        }

        // Leaf node
        if (node.triCount > 0u) {
            var i = node.firstTri;
            let end = i + node.triCount;
            let safeEnd = min(end, numTris);

            loop {
                if (i >= safeEnd) {
                    break;
                }

                let tri = getTriangle(i);
                let t   = intersectRayTriangle(ray, tri);

                if (t > 0.0 && t < closestT) {
                    closestT = t;
                    hitFound = true;

                    let edge1 = tri.v1 - tri.v0;
                    let edge2 = tri.v2 - tri.v0;
                    hitNormal = normalize(cross(edge1, edge2));
                }
                i = i + 1u;
            }
        } else {
            // Internal node → push up to 8 children, sorted front-to-back

            let firstChild = nodeIndex * 8u + 1u;

            var childData: array<vec2<f32>, 8>;
            var numValidChildren: u32 = 0u;

            for (var c = 0u; c < 8u; c = c + 1u) {
                let ci0 = firstChild + c;
                if (ci0 < numNodes) {
                    let c0 = getBVHNode(ci0);
                    let d0 = intersectAABB(ray, invDir, c0.min, c0.max, closestT);
                    if (d0 < 1e30) {
                        childData[numValidChildren] = vec2<f32>(d0, f32(ci0));
                        numValidChildren = numValidChildren + 1u;
                    }
                }
            }

            // Sort valid children by distance (Insertion sort – cheap for N ≤ 8)
            var i: u32 = 1u;
            loop {
                if (i >= numValidChildren) {
                    break;
                }
                let key = childData[i];
                var j: i32 = i32(i) - 1;
                loop {
                    if (j < 0) {
                        break;
                    }
                    if (childData[u32(j)].x <= key.x) {
                        break;
                    }
                    childData[u32(j + 1)] = childData[u32(j)];
                    j = j - 1;
                }
                childData[u32(j + 1)] = key;
                i = i + 1u;
            }

            // Push children in reverse order → closest will be popped first
            var k: i32 = i32(numValidChildren) - 1;
            loop {
                if (k < 0) {
                    break;
                }

                let ci = u32(childData[u32(k)].y);

                sp = sp + 1;
                if (u32(sp) < MAX_STACK_SIZE) {
                    stack[u32(sp)] = ci;
                } else {
                    sp = sp - 1;
                    break;
                }

                k = k - 1;
            }
        }
    }

    if (!hitFound) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }

    // Shading (simple diffuse + ambient, modulated by tris for debug)
    let lightDir = normalize(vec3<f32>(1.0, 1.5, -1.0));
    let baseColor = vec3<f32>(0.9, 0.7, 0.3);

    let finalNormal = select(
        hitNormal,
        -hitNormal,
        dot(hitNormal, ray.dir) > 0.0
    );

    let ndotl = max(dot(finalNormal, lightDir), 0.0);

    // Debug complexity factor – replace tris with nodes if you prefer node cost
    let complexity = vec3(clamp(0.01 * f32(tris), 0.0, 1.0));

    let lit = fma(baseColor, vec3(ndotl), baseColor * 0.15);

    return vec4<f32>(lit, closestT);
}

// ----------------------------------
// main()
// ----------------------------------
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res: vec2<u32> = vec2<u32>(
        u32(ubo.resolution.x),
        u32(ubo.resolution.y)
    );

    if (gid.x >= res.x || gid.y >= res.y) {
        return;
    }

    let numTris: u32 = u32(ubo.camPosNumTris.w);
    let camPos: vec3<f32> = ubo.camPosNumTris.xyz;
    let camQuat: vec4<f32> = ubo.camQuat;

    // Read node count from BVH[0]
    let numNodes: u32 = u32(BVH[0u]);

    if (numNodes == 0u || numTris == 0u) {
        textureStore(
            outputTex,
            vec2<i32>(i32(gid.x), i32(gid.y)),
            vec4<f32>(0.01, 0.01, 0.01, 1.0)
        );
        return;
    }

    let p: vec2<f32> = (vec2(f32(gid.x), f32(gid.y)) + 0.5) / vec2<f32>(f32(res.x), f32(res.y)) * 2.0 - 1.0;

    // NDC coordinates
    let px: f32 = p.x;
    let py: f32 = p.y;

    // Camera projection and direction
    let aspect: f32 = ubo.resolution.w;
    let focal: f32 = ubo.resolution.z;

    var dirCamera: vec3<f32> = normalize(vec3<f32>(
        px * aspect,
        py,
        -focal
    ));

    dirCamera = rotateVectorByQuat(dirCamera, camQuat);

    let ray = Ray(
        camPos,
        dirCamera,
        1.0 / dirCamera
    );

    let hit: vec4<f32> = traverseBVH(ray, numTris);

    var color: vec3<f32>;
    if (hit.w > 0.0) {
        color = hit.xyz;
    } else {
        color = vec3<f32>(0.01, 0.01, 0.01);
    }

    textureStore(
        outputTex,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(color, 1.0)
    );
}
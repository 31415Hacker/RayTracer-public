// ============================================================
// Single-ray BVH4 renderer (NO packets)
// SAH-friendly traversal
// ============================================================

const STACK_MAX: u32 = 64u;
const INF: f32 = 1e30;
const EPS_DIR: f32 = 1e-8;
const EPS_TRI: f32 = 1e-7;
const INVALID_U32: u32 = 0xFFFFFFFFu;

// ============================================================
// Structs
// ============================================================

struct RendererUBO {
    resolution: vec4<f32>,      // x: width, y: height, z: focal, w: aspect
    camPosNumTris: vec4<f32>,   // xyz: camera pos, w: num triangles
    camQuat: vec4<f32>,
    frameCounter: vec4<f32>,
};

struct Triangle {
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
};

struct BVHNode {
    min: vec3<f32>,
    max: vec3<f32>,
    triIndex: u32,
    count: u32,
    isLeaf: u32,
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

@group(0) @binding(4)
var<storage, read> BVHChildren: array<u32>;

// ============================================================
// Helpers
// ============================================================

fn rotateVectorByQuat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    let uv = cross(u, v);
    let uuv = cross(u, uv);
    return fma(vec3<f32>(2.0), fma(vec3<f32>(s), uv, uuv), v);
}

fn invDirOrZero(d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        select(0.0, 1.0 / d.x, abs(d.x) > EPS_DIR),
        select(0.0, 1.0 / d.y, abs(d.y) > EPS_DIR),
        select(0.0, 1.0 / d.z, abs(d.z) > EPS_DIR)
    );
}

// ============================================================
// Data access
// ============================================================

fn getTriangle(index: u32) -> Triangle {
    let b = index * 9u;
    return Triangle(
        vec3<f32>(triangles[b + 0u], triangles[b + 1u], triangles[b + 2u]),
        vec3<f32>(triangles[b + 3u], triangles[b + 4u], triangles[b + 5u]),
        vec3<f32>(triangles[b + 6u], triangles[b + 7u], triangles[b + 8u])
    );
}

fn getBVHNode(index: u32) -> BVHNode {
    let base = 2u + (index << 2u);

    let a = unpack2x16float(BVH[base + 0u]);
    let b = unpack2x16float(BVH[base + 1u]);
    let c = unpack2x16float(BVH[base + 2u]);

    let mn = vec3<f32>(a.xy, b.x);
    let mx = vec3<f32>(b.y, c.xy);

    let bits = BVH[base + 3u];
    let triIndex = bits >> 4u;
    let isLeaf = (bits >> 3u) & 1u;
    let count = bits & 7u;

    return BVHNode(mn, mx, triIndex, count, isLeaf);
}

// ============================================================
// AABB / triangle intersection
// ============================================================

fn intersectAABB(
    ro: vec3<f32>,
    rd: vec3<f32>,
    invd: vec3<f32>,
    mn: vec3<f32>,
    mx: vec3<f32>,
    tMax: f32
) -> f32 {
    var tmin = 0.0;
    var tmax = tMax;

    for (var a: u32 = 0u; a < 3u; a += 1u) {
        let o = ro[a];
        let d = rd[a];
        let inv = invd[a];

        if (abs(d) <= EPS_DIR) {
            if (o < mn[a] || o > mx[a]) { return INF; }
        } else {
            let t1 = (mn[a] - o) * inv;
            let t2 = (mx[a] - o) * inv;
            let tn = min(t1, t2);
            let tf = max(t1, t2);
            tmin = max(tmin, tn);
            tmax = min(tmax, tf);
            if (tmax < tmin) { return INF; }
        }
    }

    return tmin;
}

fn intersectTriangle(
    ro: vec3<f32>,
    rd: vec3<f32>,
    tri: Triangle,
    tBest: f32
) -> vec4<f32> {
    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;

    let p = cross(rd, e2);
    let det = dot(e1, p);
    if (abs(det) < EPS_TRI) { return vec4<f32>(INF); }

    let invDet = 1.0 / det;
    let s = ro - tri.v0;
    let u = invDet * dot(s, p);
    if (u < 0.0 || u > 1.0) { return vec4<f32>(INF); }

    let q = cross(s, e1);
    let v = invDet * dot(rd, q);
    if (v < 0.0 || (u + v) > 1.0) { return vec4<f32>(INF); }

    let t = invDet * dot(e2, q);
    if (t > EPS_TRI && t < tBest) {
        let n = cross(e1, e2);
        return vec4<f32>(n, t);
    }

    return vec4<f32>(INF);
}

// ============================================================
// BVH traversal (single ray)
// ============================================================

fn traverseBVH(ro: vec3<f32>, rd: vec3<f32>, invd: vec3<f32>, numTris: u32) -> vec4<f32> {
    let numNodes = BVH[0u];
    let root = BVH[1u];
    if (numNodes == 0u || root >= numNodes) {
        return vec4<f32>(0.0);
    }

    var stack: array<u32, STACK_MAX>;
    var sp: i32 = 0;
    stack[0] = root;

    var bestT: f32 = INF;
    var bestN: vec3<f32> = vec3<f32>(0.0);

    loop {
        if (sp < 0) { break; }

        let ni = stack[u32(sp)];
        sp -= 1;

        let node = getBVHNode(ni);
        let tNode = intersectAABB(ro, rd, invd, node.min, node.max, bestT);
        if (tNode >= bestT) { continue; }

        if (node.isLeaf != 0u) {
            var ti = node.triIndex;
            let end = min(ti + node.count, numTris);
            loop {
                if (ti >= end) { break; }
                let hit = intersectTriangle(ro, rd, getTriangle(ti), bestT);
                if (hit.w < bestT) {
                    bestT = hit.w;
                    bestN = hit.xyz;
                }
                ti += 1u;
            }
        } else {
            let base = ni * 4u;

            var cIdx: array<u32, 4>;
            var cDist: array<f32, 4>;
            var cCount: u32 = 0u;

            for (var c: u32 = 0u; c < 4u; c += 1u) {
                let ci = BVHChildren[base + c];
                if (ci == INVALID_U32 || ci >= numNodes) { continue; }
                let cn = getBVHNode(ci);
                let td = intersectAABB(ro, rd, invd, cn.min, cn.max, bestT);
                if (td < bestT) {
                    cIdx[cCount] = ci;
                    cDist[cCount] = td;
                    cCount += 1u;
                }
            }

            for (var i: u32 = 1u; i < cCount; i += 1u) {
                let kI = cIdx[i];
                let kD = cDist[i];
                var j: i32 = i32(i) - 1;
                loop {
                    if (j < 0 || cDist[u32(j)] <= kD) { break; }
                    cIdx[u32(j + 1)] = cIdx[u32(j)];
                    cDist[u32(j + 1)] = cDist[u32(j)];
                    j -= 1;
                }
                cIdx[u32(j + 1)] = kI;
                cDist[u32(j + 1)] = kD;
            }

            for (var i: i32 = i32(cCount) - 1; i >= 0; i -= 1) {
                if (sp + 1 < i32(STACK_MAX)) {
                    sp += 1;
                    stack[u32(sp)] = cIdx[u32(i)];
                }
            }
        }
    }

    return vec4<f32>(bestN, bestT);
}

// ============================================================
// Shading
// ============================================================

fn shade(n: vec3<f32>) -> vec3<f32> {
    let l = normalize(vec3<f32>(1.0, 1.5, 1.0));
    let base = vec3<f32>(0.9, 0.7, 0.3);
    return base * (0.15 + max(dot(normalize(n), l), 0.0));
}

// ============================================================
// Main
// ============================================================

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<u32>(u32(ubo.resolution.x), u32(ubo.resolution.y));
    if (gid.x >= res.x || gid.y >= res.y) { return; }

    let aspect = ubo.resolution.w;
    let focal = ubo.resolution.z;

    let q = ubo.camQuat;
    let camR = rotateVectorByQuat(vec3<f32>(1.0, 0.0, 0.0), q);
    let camU = rotateVectorByQuat(vec3<f32>(0.0, 1.0, 0.0), q);
    let camF = rotateVectorByQuat(vec3<f32>(0.0, 0.0, -1.0), q);

    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
           / vec2<f32>(f32(res.x), f32(res.y));
    let p = fma(uv, vec2<f32>(2.0), vec2<f32>(-1.0));

    let rd = normalize(camR * (p.x * aspect) + camU * p.y + camF * focal);
    let ro = ubo.camPosNumTris.xyz;
    let invd = invDirOrZero(rd);

    let hit = traverseBVH(ro, rd, invd, u32(ubo.camPosNumTris.w));

    let col = select(vec3<f32>(0.01), shade(hit.xyz), hit.w < INF);
    textureStore(outputTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(col, 1.0));
}
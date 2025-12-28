// renderer.wgsl
// BVH4 traversal (explicit 4 children, LBVH4 after CPU collapse)

const PACKET_W: u32 = 2u;
const PACKET_H: u32 = 2u;
const PACKET_SIZE: u32 = PACKET_W * PACKET_H;

const STACK_MAX: u32 = 64u;

const NODE4_STRIDE_U32: u32 = 8u;
const LEAF_FLAG: u32 = 0x80000000u;
const INVALID: u32 = 0xFFFFFFFFu;

struct RendererUBO {
    resolution: vec4<f32>,
    camPosNumTris: vec4<f32>,
    camQuat: vec4<f32>,
    frameCounter: vec4<f32>,
};

struct RayPacket {
    origin: array<vec3<f32>, PACKET_SIZE>,
    dir: array<vec3<f32>, PACKET_SIZE>,
    invdir: array<vec3<f32>, PACKET_SIZE>,
};

struct Triangle {
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
};

struct BVHNode4 {
    min: vec3<f32>,
    max: vec3<f32>,
    c0: u32,
    c1: u32,
    c2: u32,
    c3: u32,
    triIndex: u32,
    isLeaf: bool,
};

struct HitPacket {
    t: array<f32, PACKET_SIZE>,
    normal: array<vec3<f32>, PACKET_SIZE>,
    hit: array<bool, PACKET_SIZE>,
};

alias LaneMask = array<bool, PACKET_SIZE>;

@group(0) @binding(0)
var outputTex: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var<uniform> ubo: RendererUBO;

@group(0) @binding(2)
var<storage, read> triangles: array<f32>;

@group(0) @binding(3)
var<storage, read> BVH: array<u32>;

const INF: f32 = 1e30;

fn rotateVectorByQuat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    let uv = cross(u, v);
    let uuv = cross(u, uv);
    return fma(vec3<f32>(2.0), fma(vec3<f32>(s), uv, uuv), v);
}

fn safeInvDir(d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        select(INF, 1.0 / d.x, abs(d.x) > 1e-8),
        select(INF, 1.0 / d.y, abs(d.y) > 1e-8),
        select(INF, 1.0 / d.z, abs(d.z) > 1e-8)
    );
}

fn getTriangle(index: u32) -> Triangle {
    let b = index * 9u;
    return Triangle(
        vec3<f32>(triangles[b + 0u], triangles[b + 1u], triangles[b + 2u]),
        vec3<f32>(triangles[b + 3u], triangles[b + 4u], triangles[b + 5u]),
        vec3<f32>(triangles[b + 6u], triangles[b + 7u], triangles[b + 8u])
    );
}

fn getBVHNode4(index: u32) -> BVHNode4 {
    let base = 1u + index * NODE4_STRIDE_U32;

    let a = unpack2x16float(BVH[base + 0u]);
    let b = unpack2x16float(BVH[base + 1u]);
    let c = unpack2x16float(BVH[base + 2u]);

    let mn = vec3<f32>(a.x, a.y, b.x);
    let mx = vec3<f32>(b.y, c.x, c.y);

    let c0 = BVH[base + 3u];
    let c1 = BVH[base + 4u];
    let c2 = BVH[base + 5u];
    let c3 = BVH[base + 6u];

    let metadata = BVH[base + 7u];
    let isLeaf = (metadata & LEAF_FLAG) != 0u;
    let triIndex = metadata & 0x7FFFFFFFu;

    return BVHNode4(mn, mx, c0, c1, c2, c3, triIndex, isLeaf);
}

fn anyLane(mask: LaneMask) -> bool {
    var anyH: bool = false;
    for (var i: u32 = 0u; i < PACKET_SIZE; i += 1u) {
        anyH = anyH || mask[i];
    }
    return anyH;
}

fn intersectAABBPacketMask(
    packet: RayPacket,
    mn: vec3<f32>,
    mx: vec3<f32>,
    inMask: LaneMask,
    bestT: array<f32, PACKET_SIZE>,
    outMask: ptr<function, LaneMask>,
    outMinT: ptr<function, f32>
) {
    var minT: f32 = INF;
    var anyHit: bool = false;

    if (any(mn > mx)) {
        for (var i: u32 = 0u; i < PACKET_SIZE; i += 1u) {
            (*outMask)[i] = false;
        }
        (*outMinT) = INF;
        return;
    }

    for (var i: u32 = 0u; i < PACKET_SIZE; i += 1u) {
        if (!inMask[i]) {
            (*outMask)[i] = false;
            continue;
        }

        let t1 = (mn - packet.origin[i]) * packet.invdir[i];
        let t2 = (mx - packet.origin[i]) * packet.invdir[i];

        let tmin = max(
            max(min(t1.x, t2.x), min(t1.y, t2.y)),
            min(t1.z, t2.z)
        );
        let tmax = min(
            min(max(t1.x, t2.x), max(t1.y, t2.y)),
            max(t1.z, t2.z)
        );

        let hit = (tmax >= max(tmin, 0.0)) && (tmin < bestT[i]);
        (*outMask)[i] = hit;

        if (hit) {
            minT = min(minT, tmin);
            anyHit = true;
        }
    }

    (*outMinT) = select(INF, minT, anyHit);
}

fn intersectTrianglePacket(
    packet: RayPacket,
    tri: Triangle,
    triN: vec3<f32>,
    laneMask: LaneMask,
    out: ptr<function, HitPacket>
) {
    let eps = 1e-7;
    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;

    for (var i: u32 = 0u; i < PACKET_SIZE; i += 1u) {
        if (!laneMask[i]) { continue; }

        let p = cross(packet.dir[i], e2);
        let det = dot(e1, p);

        if (abs(det) < eps) { continue; }

        let invDet = 1.0 / det;
        let s = packet.origin[i] - tri.v0;
        let u = invDet * dot(s, p);

        if (u < 0.0 || u > 1.0) { continue; }

        let q = cross(s, e1);
        let v = invDet * dot(packet.dir[i], q);

        if (v < 0.0 || (u + v) > 1.0) { continue; }

        let t = invDet * dot(e2, q);
        if (t > eps && t < (*out).t[i]) {
            (*out).t[i] = t;
            (*out).normal[i] = triN;
            (*out).hit[i] = true;
        }
    }
}

fn traverseBVH4Packet(
    packet: RayPacket,
    numTris: u32,
    initMask: LaneMask
) -> HitPacket {
    let numNodes: u32 = BVH[0u];

    var out: HitPacket;
    for (var i: u32 = 0u; i < PACKET_SIZE; i += 1u) {
        out.t[i] = INF;
        out.normal[i] = vec3<f32>(0.0);
        out.hit[i] = false;
    }

    if (numNodes == 0u || numTris == 0u || !anyLane(initMask)) {
        return out;
    }

    var stack: array<u32, STACK_MAX>;
    var stackMask: array<LaneMask, STACK_MAX>;
    var sp: i32 = 0;

    stack[0] = 0u;
    stackMask[0] = initMask;

    loop {
        if (sp < 0) { break; }

        let nodeIndex = stack[u32(sp)];
        let laneMask = stackMask[u32(sp)];
        sp -= 1;

        let node = getBVHNode4(nodeIndex);

        if (any(node.min > node.max)) {
            continue;
        }

        var hitMask: LaneMask;
        var nodeMinT: f32;

        intersectAABBPacketMask(
            packet,
            node.min,
            node.max,
            laneMask,
            out.t,
            &hitMask,
            &nodeMinT
        );

        if (!anyLane(hitMask)) {
            continue;
        }

        if (node.isLeaf) {
            let ti = node.triIndex;
            if (ti < numTris) {
                let tri = getTriangle(ti);
                let triN = normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                intersectTrianglePacket(packet, tri, triN, hitMask, &out);
            }
            continue;
        }

        // BVH4 children
        var childIdx: array<u32, 4>;
        childIdx[0u] = node.c0;
        childIdx[1u] = node.c1;
        childIdx[2u] = node.c2;
        childIdx[3u] = node.c3;

        var childDist: array<f32, 4>;
        var childMasks: array<LaneMask, 4>;
        var childCount: u32 = 0u;

        for (var c: u32 = 0u; c < 4u; c += 1u) {
            let ci = childIdx[c];
            if (ci == INVALID || ci >= numNodes) { continue; }

            let child = getBVHNode4(ci);
            if (any(child.min > child.max)) { continue; }

            var cmask: LaneMask;
            var cminT: f32;

            intersectAABBPacketMask(
                packet,
                child.min,
                child.max,
                hitMask,
                out.t,
                &cmask,
                &cminT
            );

            if (anyLane(cmask)) {
                childIdx[childCount] = ci;
                childDist[childCount] = cminT;
                childMasks[childCount] = cmask;
                childCount += 1u;
            }
        }

        // select nearest child (min childDist)
        var best: u32 = 0u;
        for (var i: u32 = 1u; i < childCount; i += 1u) {
            best = select(best, i, childDist[i] < childDist[best]);
        }

        // swap nearest into slot 0
        if (best != 0u) {
            let ti = childIdx[0u];
            let td = childDist[0u];
            let tm = childMasks[0u];

            childIdx[0u]   = childIdx[best];
            childDist[0u]  = childDist[best];
            childMasks[0u] = childMasks[best];

            childIdx[best]   = ti;
            childDist[best]  = td;
            childMasks[best] = tm;
        }

        // push far -> near
        for (var i: i32 = i32(childCount) - 1; i >= 0; i -= 1) {
            if (sp + 1 < i32(STACK_MAX)) {
                sp += 1;
                stack[u32(sp)] = childIdx[u32(i)];
                stackMask[u32(sp)] = childMasks[u32(i)];
            }
        }
    }

    return out;
}

fn shade(n: vec3<f32>) -> vec3<f32> {
    let lightDir = normalize(vec3<f32>(1.0, 1.5, 1.0));
    let baseColor = vec3<f32>(0.9, 0.7, 0.3);
    let ndotl = max(dot(n, lightDir), 0.0);
    return baseColor * (0.15 + ndotl);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<u32>(u32(ubo.resolution.x), u32(ubo.resolution.y));

    let packetBase = gid.xy * vec2<u32>(PACKET_W, PACKET_H);
    if (packetBase.x >= res.x || packetBase.y >= res.y) {
        return;
    }

    let aspect = ubo.resolution.w;
    let focal = ubo.resolution.z;

    var packet: RayPacket;
    var laneMask: LaneMask;

    for (var i: u32 = 0u; i < PACKET_SIZE; i += 1u) {
        let ox = i % PACKET_W;
        let oy = i / PACKET_W;

        let px = packetBase.x + ox;
        let py = packetBase.y + oy;

        let inBounds = (px < res.x) && (py < res.y);
        laneMask[i] = inBounds;

        if (!inBounds) {
            packet.origin[i] = vec3<f32>(0.0);
            packet.dir[i] = vec3<f32>(0.0, 0.0, -1.0);
            packet.invdir[i] = vec3<f32>(INF);
            continue;
        }

        let uv = (vec2<f32>(vec2<u32>(px, py)) + 0.5) / vec2<f32>(res);
        let p = fma(uv, vec2<f32>(2.0), vec2<f32>(-1.0));

        var dir = normalize(vec3<f32>(p.x * aspect, p.y, -focal));
        dir = rotateVectorByQuat(dir, ubo.camQuat);

        packet.origin[i] = ubo.camPosNumTris.xyz;
        packet.dir[i] = dir;
        packet.invdir[i] = safeInvDir(dir);
    }

    let numTris = u32(ubo.camPosNumTris.w);
    let hits = traverseBVH4Packet(packet, numTris, laneMask);

    for (var i: u32 = 0u; i < PACKET_SIZE; i += 1u) {
        if (!laneMask[i]) { continue; }

        let ox = i % PACKET_W;
        let oy = i / PACKET_W;

        let px = packetBase.x + ox;
        let py = packetBase.y + oy;

        let col = select(vec3<f32>(0.01), shade(hits.normal[i]), hits.hit[i]);
        textureStore(outputTex, vec2<i32>(i32(px), i32(py)), vec4<f32>(col, 1.0));
    }
}
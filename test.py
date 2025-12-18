import json
import struct
import numpy as np
from pygltflib import GLTF2
from numba import njit

# ============================================================
# Config
# ============================================================

BVH_FILE = "data/BVH_full.json"
GLB_FILE = "public/assets/dragon.glb"

RAY_ORIGIN = np.array([0.0, 0.0, 2.5], dtype=np.float32)
RAY_DIR = np.array([0.0, 0.0, 1.0], dtype=np.float32)
RAY_DIR /= np.linalg.norm(RAY_DIR)

INF = np.float32(1e30)

# ============================================================
# FP16 unpack (WGSL compatible, Numba-safe)
# ============================================================
def unpack2x16float(u32):
    lo = np.uint16(u32 & 0xFFFF)
    hi = np.uint16((u32 >> 16) & 0xFFFF)

    f0 = np.float32(np.float16(lo))
    f1 = np.float32(np.float16(hi))
    return f0, f1

# ============================================================
# BVH node decode (FIXED return signature)
# ============================================================

def get_bvh_node(BVH, index):
    base = 1 + index * 4

    if base + 3 >= BVH.shape[0]:
        return (
            False,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0, 0
        )

    a0, a1 = unpack2x16float(BVH[base + 0])
    b0, b1 = unpack2x16float(BVH[base + 1])
    c0, c1 = unpack2x16float(BVH[base + 2])

    mnx, mny, mnz = a0, a1, b0
    mxx, mxy, mxz = b1, c0, c1

    bits = BVH[base + 3]
    firstTri = bits >> 3
    triCount = bits & 7

    valid = (mnx <= mxx) and (mny <= mxy) and (mnz <= mxz)

    return (
        valid,
        mnx, mny, mnz,
        mxx, mxy, mxz,
        firstTri,
        triCount
    )

# ============================================================
# Ray / AABB
# ============================================================

@njit
def intersect_aabb(ro, invrd, mnx, mny, mnz, mxx, mxy, mxz):
    t1x = (mnx - ro[0]) * invrd[0]
    t2x = (mxx - ro[0]) * invrd[0]
    t1y = (mny - ro[1]) * invrd[1]
    t2y = (mxy - ro[1]) * invrd[1]
    t1z = (mnz - ro[2]) * invrd[2]
    t2z = (mxz - ro[2]) * invrd[2]

    tmin = max(min(t1x, t2x), min(t1y, t2y), min(t1z, t2z))
    tmax = min(max(t1x, t2x), max(t1y, t2y), max(t1z, t2z))

    if tmax >= max(tmin, 0.0):
        return tmin
    return INF

# ============================================================
# Ray / Triangle (Möller–Trumbore)
# ============================================================

@njit
def intersect_triangle(ro, rd, v0, v1, v2):
    eps = 1e-7

    e1 = v1 - v0
    e2 = v2 - v0
    p = np.cross(rd, e2)
    det = np.dot(e1, p)

    if abs(det) < eps:
        return INF

    invDet = 1.0 / det
    s = ro - v0
    u = invDet * np.dot(s, p)
    if u < 0.0 or u > 1.0:
        return INF

    q = np.cross(s, e1)
    v = invDet * np.dot(rd, q)
    if v < 0.0 or u + v > 1.0:
        return INF

    t = invDet * np.dot(e2, q)
    return t if t > eps else INF

# ============================================================
# BVH4 traversal (near-first)
# ============================================================

def traverse_bvh(BVH, triangles, ro, rd):
    num_nodes = BVH[0]

    invrd = np.empty(3, np.float32)
    for i in range(3):
        invrd[i] = 1.0 / rd[i] if abs(rd[i]) > 1e-8 else INF

    stack = np.empty(64, np.uint32)
    sp = 0
    stack[0] = 0

    closest_t = INF
    hit_tri = -1
    visited = 0
    leaf_hits = 0

    while sp >= 0:
        node = stack[sp]
        sp -= 1
        visited += 1

        (
            ok,
            mnx, mny, mnz,
            mxx, mxy, mxz,
            firstTri,
            triCount
        ) = get_bvh_node(BVH, node)

        if not ok:
            continue

        d = intersect_aabb(ro, invrd, mnx, mny, mnz, mxx, mxy, mxz)
        if d >= closest_t:
            continue

        if triCount > 0:
            leaf_hits += 1
            for i in range(firstTri, firstTri + triCount):
                v0 = triangles[i, 0]
                v1 = triangles[i, 1]
                v2 = triangles[i, 2]
                t = intersect_triangle(ro, rd, v0, v1, v2)
                if t < closest_t:
                    closest_t = t
                    hit_tri = i
        else:
            base = node * 4 + 1
            for c in range(4):
                ci = base + c
                if ci < num_nodes:
                    sp += 1
                    stack[sp] = ci

    return visited, leaf_hits, hit_tri, closest_t

# ============================================================
# GLB loader (Scene.js equivalent)
# ============================================================

def load_glb_triangles(path):
    gltf = GLTF2().load(path)
    blob = gltf.binary_blob()

    tris = []

    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            acc = gltf.accessors[prim.attributes.POSITION]
            view = gltf.bufferViews[acc.bufferView]

            offset = (view.byteOffset or 0) + (acc.byteOffset or 0)
            raw = blob[offset : offset + acc.count * 12]
            verts = np.frombuffer(raw, dtype=np.float32).reshape((-1, 3))

            if prim.indices is not None:
                iacc = gltf.accessors[prim.indices]
                iview = gltf.bufferViews[iacc.bufferView]
                ioff = (iview.byteOffset or 0) + (iacc.byteOffset or 0)
                iraw = blob[ioff : ioff + iacc.count * 4]
                indices = np.frombuffer(iraw, dtype=np.uint32)
            else:
                indices = np.arange(len(verts), dtype=np.uint32)

            for i in range(0, len(indices), 3):
                tris.append((
                    verts[indices[i + 0]],
                    verts[indices[i + 1]],
                    verts[indices[i + 2]],
                ))

    tris = np.array(tris, dtype=np.float32)
    normalize_mesh(tris)
    return tris

# ============================================================
# Normalize mesh (matches Scene.js)
# ============================================================

def normalize_mesh(tris):
    mn = np.min(tris.reshape(-1, 3), axis=0)
    mx = np.max(tris.reshape(-1, 3), axis=0)
    center = (mn + mx) * 0.5
    scale = 2.0 / np.max(mx - mn)
    tris[:] = (tris - center) * scale
    print("Mesh normalized.")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Loading BVH...")
    with open(BVH_FILE) as f:
        BVH = np.array(json.load(f), dtype=np.uint32)
    print("BVH nodes:", BVH[0])

    print("Loading GLB triangles...")
    triangles = load_glb_triangles(GLB_FILE)
    print("Triangles:", len(triangles))

    print("Running BVH traversal (JIT warmup)...")
    traverse_bvh(BVH, triangles, RAY_ORIGIN, RAY_DIR)

    visited, leaf_hits, hit_tri, t = traverse_bvh(
        BVH, triangles, RAY_ORIGIN, RAY_DIR
    )

    print("\n=== BVH DEBUG TRAVERSAL END ===")
    print("Visited nodes :", visited)
    print("Leaf hits     :", leaf_hits)
    print("Triangle hit  :", hit_tri)
    print("Closest t     :", t)
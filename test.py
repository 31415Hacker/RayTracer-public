import json
import numpy as np
from pygltflib import GLTF2

INF = np.float32(1e30)

BVH_FILE = "data/BVH_full.json"
GLB_FILE = "public/assets/dragon.glb"

RAY_ORIGIN = np.array([0.0, 0.0, 2.5], dtype=np.float32)
RAY_DIR = np.array([0.0, 0.0, -1.0], dtype=np.float32)
RAY_DIR /= np.linalg.norm(RAY_DIR)

NODES_INTERSECTED = 0

# ============================================================
# FP16 unpack
# ============================================================

def unpack2x16float(u32):
    lo = np.uint16(u32 & 0xFFFF)
    hi = np.uint16((u32 >> 16) & 0xFFFF)

    f0 = np.frombuffer(lo.tobytes(), dtype=np.float16)[0]
    f1 = np.frombuffer(hi.tobytes(), dtype=np.float16)[0]

    return np.float32(f0), np.float32(f1)

# ============================================================
# BVH node decode
# ============================================================

def get_bvh_node(BVH, index):
    base = 1 + index * 4
    if base + 3 >= BVH.shape[0]:
        return None

    a0, a1 = unpack2x16float(BVH[base + 0])
    b0, b1 = unpack2x16float(BVH[base + 1])
    c0, c1 = unpack2x16float(BVH[base + 2])

    mn = np.array([a0, a1, b0], dtype=np.float32)
    mx = np.array([b1, c0, c1], dtype=np.float32)

    bits = BVH[base + 3]
    firstTri = bits >> 3
    triCount = bits & 7

    valid = np.all(mn <= mx)

    return {
        "index": index,
        "mn": mn,
        "mx": mx,
        "firstTri": firstTri,
        "triCount": triCount,
        "valid": valid
    }

# ============================================================
# Ray / AABB
# ============================================================

def intersect_aabb(ro, invrd, mn, mx):
    global NODES_INTERSECTED
    NODES_INTERSECTED += 1
    t1 = (mn - ro) * invrd
    t2 = (mx - ro) * invrd

    tmin = max(min(t1[0], t2[0]), min(t1[1], t2[1]), min(t1[2], t2[2]))
    tmax = min(max(t1[0], t2[0]), max(t1[1], t2[1]), max(t1[2], t2[2]))

    if tmax >= max(tmin, 0.0):
        return np.float32(tmin)
    return INF

# ============================================================
# Ray / Triangle
# ============================================================

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
# GLB loader
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
                    verts[indices[i]],
                    verts[indices[i+1]],
                    verts[indices[i+2]],
                ))

    tris = np.array(tris, dtype=np.float32)
    normalize_mesh(tris)
    return tris

def normalize_mesh(tris):
    mn = np.min(tris.reshape(-1, 3), axis=0)
    mx = np.max(tris.reshape(-1, 3), axis=0)
    center = (mn + mx) * 0.5
    scale = 2.0 / np.max(mx - mn)
    tris[:] = (tris - center) * scale

# ============================================================
# FULL DEBUG BVH TRAVERSAL
# ============================================================

def traverse_bvh_debug(BVH, triangles, ro, rd):
    num_nodes = int(BVH[0])
    invrd = np.array([
        1.0 / rd[i] if abs(rd[i]) > 1e-8 else INF
        for i in range(3)
    ], dtype=np.float32)

    stack = []
    stack.append(0)

    closest_t = INF
    hit_tri = -1
    step = 0

    print("\n===== BVH DEBUG TRACE START =====\n")

    while stack:
        node = stack.pop()
        step += 1

        print(f"\n--- STEP {step} ---")
        print("POP node:", node)
        print("STACK BEFORE:", stack)

        n = get_bvh_node(BVH, node)
        print("Node: ", n, "\n")

        if n is None or not n["valid"]:
            print("Node invalid → skip")
            continue

        tmin = intersect_aabb(ro, invrd, n["mn"], n["mx"])
        print("AABB tmin:", tmin, "closest_t:", closest_t)

        if tmin >= closest_t:
            print("AABB rejected")
            continue

        if n["triCount"] > 0:
            print("LEAF node")
            for i in range(n["firstTri"], n["firstTri"] + n["triCount"]):
                t = intersect_triangle(ro, rd, *triangles[i])
                if t < closest_t:
                    closest_t = t
                    hit_tri = i
                    print(f"  HIT triangle {i} at t={t}")
            continue

        print("INTERNAL node")

        base = node * 4 + 1
        children = []

        for c in range(4):
            ci = base + c
            if ci >= num_nodes:
                continue

            cn = get_bvh_node(BVH, ci)
            if cn is None or not cn["valid"]:
                continue

            ct = intersect_aabb(ro, invrd, cn["mn"], cn["mx"])
            if ct < closest_t:
                children.append((ct, ci))
                print(f"  CHILD {ci} accepted, tmin={ct}")
            else:
                print(f"  CHILD {ci} rejected")

        children.sort(key=lambda x: x[0])

        for _, ci in reversed(children):
            stack.append(ci)

        print("STACK AFTER:", stack)

    print("\n===== BVH DEBUG TRACE END =====")
    print("HIT TRIANGLE:", hit_tri)
    print("CLOSEST T:", closest_t)

    global NODES_INTERSECTED
    print("Nodes intersected:", NODES_INTERSECTED)

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    with open(BVH_FILE) as f:
        BVH = np.array(json.load(f), dtype=np.uint32)

    triangles = load_glb_triangles(GLB_FILE)

    traverse_bvh_debug(BVH, triangles, RAY_ORIGIN, RAY_DIR)
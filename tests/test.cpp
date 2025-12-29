#include <cstdlib>
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cmath>

/* ============================================================
   Constants
============================================================ */

static constexpr uint32_t NODE2_STRIDE_U32 = 6;
static constexpr uint32_t NODE4_STRIDE_U32 = 8;
static constexpr uint32_t LEAF_FLAG = 0x80000000u;
static constexpr uint32_t INVALID   = 0xFFFFFFFFu;

/* ============================================================
   FP16 helpers (bit-exact with JS)
============================================================ */

static inline uint16_t f32_to_f16(float v) {
    uint32_t u = *reinterpret_cast<uint32_t*>(&v);
    uint32_t s = (u >> 16) & 0x8000;
    int32_t  e = ((u >> 23) & 0xFF) - 112;
    uint32_t m = (u >> 13) & 0x03FF;

    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7C00);
    return (uint16_t)(s | (e << 10) | m);
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t s = (h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x03FF;

    if (e == 0) {
        if (m == 0) {
            uint32_t u = s;
            return *reinterpret_cast<float*>(&u);
        }
        e = 1;
        while ((m & 0x0400) == 0) {
            m <<= 1;
            e--;
        }
        m &= 0x03FF;
    } else if (e == 31) {
        uint32_t u = s | 0x7F800000 | (m << 13);
        return *reinterpret_cast<float*>(&u);
    }

    uint32_t u = s | ((e + 112) << 23) | (m << 13);
    return *reinterpret_cast<float*>(&u);
}

static inline uint32_t pack16x2(float a, float b) {
    return uint32_t(f32_to_f16(a)) | (uint32_t(f32_to_f16(b)) << 16);
}

static inline float unpack16x2(uint32_t u, int idx) {
    return f16_to_f32((u >> (idx * 16)) & 0xFFFF);
}

/* ============================================================
   BVH4 Collapse (Iterative)
============================================================ */

struct BVH4Result {
    std::vector<uint32_t> bvh4;
    uint32_t numNodes4;
};

BVH4Result collapseLBVH2ToBVH4(
    const std::vector<uint32_t>& bvh2,
    uint32_t numTris
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    const uint32_t numNodes2 = (numTris > 0) ? (2 * numTris - 1) : 0;
    if (numNodes2 == 0) {
        return {{0}, 0};
    }

    auto node2Offset = [&](uint32_t i) {
        return 1u + i * NODE2_STRIDE_U32;
    };

    auto isLeaf2 = [&](uint32_t i) {
        return (bvh2[node2Offset(i) + 5] & LEAF_FLAG) != 0;
    };

    auto getChildren2 = [&](uint32_t i) {
        uint32_t off = node2Offset(i);
        return std::pair<uint32_t,uint32_t>(
            bvh2[off + 3], bvh2[off + 4]
        );
    };

    auto getBoundsPacked2 = [&](uint32_t i) {
        uint32_t off = node2Offset(i);
        return std::array<uint32_t,3>{
            bvh2[off + 0],
            bvh2[off + 1],
            bvh2[off + 2]
        };
    };

    auto getMeta2 = [&](uint32_t i) {
        return bvh2[node2Offset(i) + 5];
    };

    struct Bounds {
        float min[3];
        float max[3];
    };

    auto decodeBounds = [&](uint32_t b0, uint32_t b1, uint32_t b2) {
        Bounds b;
        b.min[0] = unpack16x2(b0, 0);
        b.min[1] = unpack16x2(b0, 1);
        b.min[2] = unpack16x2(b1, 0);
        b.max[0] = unpack16x2(b1, 1);
        b.max[1] = unpack16x2(b2, 0);
        b.max[2] = unpack16x2(b2, 1);
        return b;
    };

    auto encodeBounds = [&](const Bounds& b) {
        return std::array<uint32_t,3>{
            pack16x2(b.min[0], b.min[1]),
            pack16x2(b.min[2], b.max[0]),
            pack16x2(b.max[1], b.max[2])
        };
    };

    std::vector<uint32_t> out;
    out.reserve(numNodes2 * NODE4_STRIDE_U32);
    out.push_back(0);

    auto emitNode4 = [&]() {
        uint32_t idx = (uint32_t)((out.size() - 1) / NODE4_STRIDE_U32);
        out.resize(out.size() + NODE4_STRIDE_U32, 0);
        return idx;
    };

    auto writeNode4 = [&](uint32_t idx,
                          uint32_t b0, uint32_t b1, uint32_t b2,
                          uint32_t c0, uint32_t c1,
                          uint32_t c2, uint32_t c3,
                          uint32_t meta) {
        uint32_t base = 1 + idx * NODE4_STRIDE_U32;
        out[base + 0] = b0;
        out[base + 1] = b1;
        out[base + 2] = b2;
        out[base + 3] = c0;
        out[base + 4] = c1;
        out[base + 5] = c2;
        out[base + 6] = c3;
        out[base + 7] = meta;
    };

    struct Frame {
        uint32_t node2;
        uint32_t node4;
        bool expanded;

        uint32_t kids[4];
        uint32_t kidCount;

        uint32_t child4[4];
        uint32_t parentChildSlot;
    };

    std::vector<Frame> stack;
    stack.reserve(numNodes2);

    stack.push_back({
        0,
        INVALID,
        false,
        {},
        0,
        {INVALID, INVALID, INVALID, INVALID},
        INVALID
    });

    while (!stack.empty()) {
        Frame& f = stack.back();

        if (!f.expanded) {
            f.node4 = emitNode4();

            if (isLeaf2(f.node2)) {
                auto b = getBoundsPacked2(f.node2);
                writeNode4(
                    f.node4,
                    b[0], b[1], b[2],
                    INVALID, INVALID, INVALID, INVALID,
                    getMeta2(f.node2)
                );

                uint32_t finished = f.node4;
                uint32_t slot = f.parentChildSlot;
                stack.pop_back();

                if (!stack.empty() && slot != INVALID) {
                    stack.back().child4[slot] = finished;
                }
                continue;
            }

            auto [l, r] = getChildren2(f.node2);
            f.kids[0] = l;
            f.kids[1] = r;
            f.kidCount = 2;

            bool changed = true;
            while (f.kidCount < 4 && changed) {
                changed = false;
                for (uint32_t i = 0; i < f.kidCount; i++) {
                    uint32_t k = f.kids[i];
                    if (k != INVALID && !isLeaf2(k)) {
                        auto [cl, cr] = getChildren2(k);
                        f.kids[i] = f.kids[f.kidCount - 1];
                        f.kids[f.kidCount - 1] = cl;
                        f.kids[f.kidCount++] = cr;
                        changed = true;
                        break;
                    }
                }
            }

            f.expanded = true;

            for (int i = (int)f.kidCount - 1; i >= 0; i--) {
                if (f.kids[i] == INVALID) continue;

                stack.push_back({
                    f.kids[i],
                    INVALID,
                    false,
                    {},
                    0,
                    {INVALID, INVALID, INVALID, INVALID},
                    (uint32_t)i
                });
            }
        } else {
            Bounds acc;
            acc.min[0] = acc.min[1] = acc.min[2] = +INFINITY;
            acc.max[0] = acc.max[1] = acc.max[2] = -INFINITY;

            for (uint32_t i = 0; i < f.kidCount; i++) {
                uint32_t ci = f.child4[i];
                if (ci == INVALID) continue;
                uint32_t base = 1 + ci * NODE4_STRIDE_U32;
                Bounds cb = decodeBounds(
                    out[base],
                    out[base + 1],
                    out[base + 2]
                );
                for (int k = 0; k < 3; k++) {
                    acc.min[k] = std::min(acc.min[k], cb.min[k]);
                    acc.max[k] = std::max(acc.max[k], cb.max[k]);
                }
            }

            auto b = encodeBounds(acc);
            writeNode4(
                f.node4,
                b[0], b[1], b[2],
                f.child4[0], f.child4[1],
                f.child4[2], f.child4[3],
                0u
            );

            uint32_t finished = f.node4;
            uint32_t slot = f.parentChildSlot;
            stack.pop_back();

            if (!stack.empty() && slot != INVALID) {
                stack.back().child4[slot] = finished;
            }
        }
    }

    uint32_t numNodes4 = (uint32_t)((out.size() - 1) / NODE4_STRIDE_U32);
    out[0] = numNodes4;

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

    printf("BVH2 → BVH4 collapse (iterative): %.3f ms\n", ms);
    printf("BVH4 nodes: %u\n", numNodes4);

    return {out, numNodes4};
}

int main() {
    // ---- Load BVH2.bin ----
    const char* filename = "data/BVH2.bin";

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open " << filename << "\n";
        return 1;
    }

    const size_t fileSize = file.tellg();
    if (fileSize % 4 != 0) {
        std::cerr << "Invalid BVH2 file size\n";
        return 1;
    }

    file.seekg(0, std::ios::beg);

    std::vector<uint32_t> bvh2U32(fileSize / 4);
    file.read(reinterpret_cast<char*>(bvh2U32.data()), fileSize);
    file.close();

    // ---- Sanity checks ----
    if (bvh2U32.empty()) {
        std::cerr << "BVH2 buffer empty\n";
        return 1;
    }

    const uint32_t numNodes2 = bvh2U32[0];
    if (numNodes2 == 0) {
        std::cerr << "BVH2 reports zero nodes\n";
        return 1;
    }

    // Derive numTris from LBVH2 formula
    // numNodes2 = 2*numTris - 1
    const uint32_t numTris = (numNodes2 + 1) / 2;

    std::cout << "Loaded BVH2\n";
    std::cout << "  File size: " << fileSize / (1024 * 1024) << " MB\n";
    std::cout << "  numNodes2: " << numNodes2 << "\n";
    std::cout << "  numTris:   " << numTris << "\n";

    // ---- Collapse ----
    auto result = collapseLBVH2ToBVH4(bvh2U32, numTris);

    std::cout << "Collapse finished\n";
    std::cout << "BVH4 nodes: " << result.numNodes4 << "\n";

    return 0;
}
#include <cstdint>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <queue>

static constexpr uint32_t NODE2_STRIDE_U32 = 6;
static constexpr uint32_t NODE4_STRIDE_U32 = 8;
static constexpr uint32_t LEAF_FLAG = 0x80000000u;
static constexpr uint32_t INVALID   = 0xFFFFFFFFu;

/* ================= File IO ================= */

static bool load_u32_file(const char* path, std::vector<uint32_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize size = f.tellg();
    if (size <= 0 || (size & 3)) return false;
    f.seekg(0, std::ios::beg);
    out.resize(size_t(size >> 2));
    return bool(f.read(reinterpret_cast<char*>(out.data()), size));
}

static bool save_u32_file(const char* path, const std::vector<uint32_t>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data.data()),
            std::streamsize(data.size() << 2));
    return bool(f);
}

/* ================= BVH helpers ================= */

static inline size_t node2_off(uint32_t n) {
    return size_t(1u + n * NODE2_STRIDE_U32);
}

static inline size_t node4_off(uint32_t n) {
    return size_t(1u + n * NODE4_STRIDE_U32);
}

static inline bool is_leaf2(const std::vector<uint32_t>& bvh2,
                            uint32_t n,
                            uint32_t numNodes2) {
    if (n >= numNodes2) return true;
    return (bvh2[node2_off(n) + 5] & LEAF_FLAG) != 0;
}

static void print_bvh4_first_depth3(
    const std::vector<uint32_t>& bvh4,
    uint32_t numNodes4
) {
    struct Item {
        uint32_t node;
        uint32_t depth;
    };

    std::queue<Item> q;
    q.push({0, 0}); // root

    std::cout << "\n=== BVH4 first depth-3 nodes ===\n";

    while (!q.empty()) {
        Item it = q.front();
        q.pop();

        if (it.node >= numNodes4) continue;

        size_t off = node4_off(it.node);
        uint32_t meta = bvh4[off + 7];

        std::cout
            << "Node " << it.node
            << " | depth " << it.depth
            << " | " << ((meta & LEAF_FLAG) ? "LEAF" : "INTERNAL")
            << " | kids: ";

        // print children
        for (int i = 0; i < 4; i++) {
            uint32_t c = bvh4[off + 3 + i];
            if (c != INVALID) std::cout << c << " ";
        }
        std::cout << "\n";

        // stop expanding beyond depth 3
        if (it.depth == 3) continue;

        // expand only internal nodes
        if (!(meta & LEAF_FLAG)) {
            for (int i = 0; i < 4; i++) {
                uint32_t c = bvh4[off + 3 + i];
                if (c != INVALID) {
                    q.push({c, it.depth + 1});
                }
            }
        }
    }

    std::cout << "================================\n\n";
}

/* ================= BVH2 → BVH4 promotion ================= */

static inline void promote_children_4(
    const std::vector<uint32_t>& bvh2,
    uint32_t numNodes2,
    uint32_t left,
    uint32_t right,
    uint32_t out[4]
) {
    uint32_t k = 0;

    auto push = [&](uint32_t c) {
        if (k < 4) out[k++] = c;
    };

    auto promote = [&](uint32_t c) {
        if (c == INVALID) return;

        if (is_leaf2(bvh2, c, numNodes2)) {
            push(c);
        } else {
            size_t off = node2_off(c);
            push(bvh2[off + 3]); // left child
            push(bvh2[off + 4]); // right child
        }
    };

    promote(left);
    promote(right);

    while (k < 4) out[k++] = INVALID;
}

/* ================= Main ================= */

int main(int argc, char** argv) {
    const char* inPath  = "data/BVH2.bin";
    const char* outPath = "data/BVH4_wide.bin";

    if (argc > 1) inPath  = argv[1];
    if (argc > 2) outPath = argv[2];

    std::vector<uint32_t> bvh2;
    if (!load_u32_file(inPath, bvh2)) {
        std::cerr << "Failed to read BVH2\n";
        return 1;
    }

    uint32_t numNodes2 = bvh2[0];

    std::vector<uint32_t> bvh4;
    bvh4.resize(size_t(1) + size_t(numNodes2) * NODE4_STRIDE_U32);
    bvh4[0] = numNodes2;

    uint64_t leafCount = 0;
    uint64_t internalCount = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint32_t n = 0; n < numNodes2; ++n) {
        size_t o2 = node2_off(n);
        size_t o4 = node4_off(n);

        // copy bounds
        bvh4[o4 + 0] = bvh2[o2 + 0];
        bvh4[o4 + 1] = bvh2[o2 + 1];
        bvh4[o4 + 2] = bvh2[o2 + 2];

        uint32_t meta = bvh2[o2 + 5];

        if (meta & LEAF_FLAG) {
            leafCount++;
            bvh4[o4 + 3] = INVALID;
            bvh4[o4 + 4] = INVALID;
            bvh4[o4 + 5] = INVALID;
            bvh4[o4 + 6] = INVALID;
            bvh4[o4 + 7] = meta;
        } else {
            internalCount++;

            uint32_t left  = bvh2[o2 + 3];
            uint32_t right = bvh2[o2 + 4];

            uint32_t kids[4];
            promote_children_4(bvh2, numNodes2, left, right, kids);

            bvh4[o4 + 3] = kids[0];
            bvh4[o4 + 4] = kids[1];
            bvh4[o4 + 5] = kids[2];
            bvh4[o4 + 6] = kids[3];
            bvh4[o4 + 7] = 0;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "BVH2 → BVH4 (O(N)) time: " << ms << " ms\n";
    std::cout << "leaves: " << leafCount << " internals: " << internalCount << "\n";

    print_bvh4_first_depth3(bvh4, numNodes2);

    save_u32_file(outPath, bvh4);
    return 0;
}
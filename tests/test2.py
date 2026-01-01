INVALID = -1

class Node:
    def __init__(self, left=INVALID, right=INVALID, is_leaf=False, triangles=None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.triangles = triangles or []
        self.children = []   # BVH4 children after collapse

    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(tris={self.triangles})"
        return f"Internal(children={self.children})"


# ------------------------------------------------------------
# Build a toy BVH2
# ------------------------------------------------------------

nodes = []

# Leaves
nodes.append(Node(is_leaf=True, triangles=["A"]))  # 0 placeholder, will not be used
nodes.append(Node(is_leaf=True, triangles=["D"]))  # 1
nodes.append(Node(is_leaf=True, triangles=["A"]))  # 2
nodes.append(Node(is_leaf=True, triangles=["B"]))  # 3
nodes.append(Node(is_leaf=True, triangles=["C"]))  # 4

# Internal nodes
nodes.append(Node(left=2, right=4))  # 5 internal
nodes.append(Node(left=3, right=5))  # 6 internal
nodes.append(Node(left=6, right=1))  # 7 root

# Rename indices for clarity
# 1: leaf D
# 2: leaf A
# 3: leaf B
# 4: leaf C
# 5: internal (A, C)
# 6: internal (B, internal)
# 7: root

nodes_at_depth = {
    0: [7],
    1: [6],
    2: [5],
}

max_depth = 2

# ------------------------------------------------------------
# BVH2 → BVH4 collapse (single block, working)
# ------------------------------------------------------------

for d in range(max_depth + 1):
    for n in nodes_at_depth.get(d, []):
        if nodes[n].is_leaf:
            continue

        queue = []
        new_children = []

        if nodes[n].left != INVALID:
            queue.append(nodes[n].left)
        if nodes[n].right != INVALID:
            queue.append(nodes[n].right)

        while queue and len(new_children) < 4:
            c = queue.pop(0)

            if c == INVALID:
                continue

            if nodes[c].is_leaf:
                new_children.append(c)
            else:
                if nodes[c].left != INVALID:
                    queue.append(nodes[c].left)
                if nodes[c].right != INVALID:
                    queue.append(nodes[c].right)

        # No leaves found → collapse to leaf
        if len(new_children) == 0:
            nodes[n].is_leaf = True
            nodes[n].children = []
            nodes[n].triangles = []

            stack = []
            if nodes[n].left != INVALID:
                stack.append(nodes[n].left)
            if nodes[n].right != INVALID:
                stack.append(nodes[n].right)

            while stack:
                c = stack.pop()
                if c == INVALID:
                    continue
                if nodes[c].is_leaf:
                    nodes[n].triangles.extend(nodes[c].triangles)
                else:
                    if nodes[c].left != INVALID:
                        stack.append(nodes[c].left)
                    if nodes[c].right != INVALID:
                        stack.append(nodes[c].right)
            continue

        nodes[n].is_leaf = False
        while len(new_children) < 4:
            new_children.append(INVALID)

        nodes[n].children = new_children

# ------------------------------------------------------------
# Inspect result
# ------------------------------------------------------------

print("After BVH2 → BVH4 collapse:\n")
for i, n in enumerate(nodes):
    print(f"Node {i}: {n}")

print("\nRoot BVH4 children and triangles:")
root = nodes[7]
for c in root.children:
    if c != INVALID:
        print(f" child {c}: {nodes[c]}")
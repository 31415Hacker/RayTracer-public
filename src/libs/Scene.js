import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

export class Scene {
    constructor() {
        this.loader = new GLTFLoader();
        this.triangles = [];
        this._normalizeEnabled = false;
        this._normalizeMode = "cube"; // "cube" | "sphere"
    }

    // ------------------------------------------------------------
    // Load GLB with normalization option
    // ------------------------------------------------------------
    async loadGLB(url, options = {}) {
        this._normalizeEnabled = options.normalize ?? false;
        this._normalizeMode = options.mode ?? "cube";

        const gltf = await new Promise((resolve, reject) => {
            this.loader.load(
                url,
                resolve,
                (xhr) =>
                    console.log(
                        `Loading: ${(xhr.loaded / xhr.total * 100).toFixed(1)}%`
                    ),
                (err) => {
                    console.error("GLB load failed:", err);
                    reject(err);
                }
            );
        });

        this.parseGLTF(gltf);

        if (this._normalizeEnabled) {
            console.log(`Normalizing mesh (${this._normalizeMode})...`);
            this.normalizeMesh();
        }

        //this.sortTriangles();
    }

    // ------------------------------------------------------------
    // Extract triangles from the GLB
    // ------------------------------------------------------------
    parseGLTF(gltf) {
        const scene = gltf.scene;
        scene.updateMatrixWorld(true);

        this.triangles.length = 0;

        scene.traverse((obj) => {
            if (!obj.isMesh) return;

            const geom = obj.geometry;
            const worldMatrix = obj.matrixWorld.clone();

            const nonIndexed =
                geom.index !== null ? geom.toNonIndexed() : geom.clone();

            const pos = nonIndexed.getAttribute("position");
            if (!pos) return;

            const array = pos.array;
            const count = pos.count;

            for (let i = 0; i < count; i += 3) {
                const v0 = new THREE.Vector3(
                    array[(i + 0) * 3],
                    array[(i + 0) * 3 + 1],
                    array[(i + 0) * 3 + 2]
                ).applyMatrix4(worldMatrix);

                const v1 = new THREE.Vector3(
                    array[(i + 1) * 3],
                    array[(i + 1) * 3 + 1],
                    array[(i + 1) * 3 + 2]
                ).applyMatrix4(worldMatrix);

                const v2 = new THREE.Vector3(
                    array[(i + 2) * 3],
                    array[(i + 2) * 3 + 1],
                    array[(i + 2) * 3 + 2]
                ).applyMatrix4(worldMatrix);

                const cx = (v0.x + v1.x + v2.x) / 3;
                const cy = (v0.y + v1.y + v2.y) / 3;
                const cz = (v0.z + v1.z + v2.z) / 3;

                this.triangles.push({
                    v0: [v0.x, v0.y, v0.z],
                    v1: [v1.x, v1.y, v1.z],
                    v2: [v2.x, v2.y, v2.z],
                    centroid: [cx, cy, cz],
                });
            }
        });
    }

    // ------------------------------------------------------------
    // Normalize mesh to [-1,1] or sphere radius 1
    // ------------------------------------------------------------
    normalizeMesh() {
        if (this.triangles.length === 0) return;

        let min = [Infinity, Infinity, Infinity];
        let max = [-Infinity, -Infinity, -Infinity];

        // Compute bounds
        for (const t of this.triangles) {
            const verts = [t.v0, t.v1, t.v2];
            for (const v of verts) {
                min[0] = Math.min(min[0], v[0]);
                min[1] = Math.min(min[1], v[1]);
                min[2] = Math.min(min[2], v[2]);

                max[0] = Math.max(max[0], v[0]);
                max[1] = Math.max(max[1], v[1]);
                max[2] = Math.max(max[2], v[2]);
            }
        }

        const center = [
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        ];

        const size = [
            max[0] - min[0],
            max[1] - min[1],
            max[2] - min[2],
        ];

        const maxDim = Math.max(size[0], size[1], size[2]);

        // cube normalization: [-1,1]
        let scale = 2.0 / maxDim;

        if (this._normalizeMode === "sphere") {
            // sphere normalization: |p| <= 1
            scale = 1.0 / (maxDim * 0.5);
        }

        // Apply normalization
        for (const t of this.triangles) {
            const arr = ["v0", "v1", "v2"];
            for (const key of arr) {
                const v = t[key];
                v[0] = (v[0] - center[0]) * scale;
                v[1] = (v[1] - center[1]) * scale;
                v[2] = (v[2] - center[2]) * scale;
            }

            // Recompute centroid
            t.centroid = [
                (t.v0[0] + t.v1[0] + t.v2[0]) / 3,
                (t.v0[1] + t.v1[1] + t.v2[1]) / 3,
                (t.v0[2] + t.v1[2] + t.v2[2]) / 3,
            ];
        }

        console.log("Mesh normalized.");
    }

    // ------------------------------------------------------------
    // Morton sorting (unchanged)
    // ------------------------------------------------------------
    sortTriangles() {
        if (this.triangles.length === 0) return;

        console.log(`Sorting ${this.triangles.length} triangles spatially...`);

        let min = [Infinity, Infinity, Infinity];
        let max = [-Infinity, -Infinity, -Infinity];

        for (const t of this.triangles) {
            for (let i = 0; i < 3; i++) {
                min[i] = Math.min(min[i], t.centroid[i]);
                max[i] = Math.max(max[i], t.centroid[i]);
            }
        }

        const bounds = [
            max[0] - min[0] || 1,
            max[1] - min[1] || 1,
            max[2] - min[2] || 1,
        ];

        const expandBits = (v) => {
            v = (v * 0x00010001) & 0xff0000ff;
            v = (v * 0x00000101) & 0x0f00f00f;
            v = (v * 0x00000011) & 0xc30c30c3;
            v = (v * 0x00000005) & 0x49249249;
            return v;
        };

        const morton3D = (x, y, z) => {
            x = Math.min(Math.max(x * 1024, 0), 1023);
            y = Math.min(Math.max(y * 1024, 0), 1023);
            z = Math.min(Math.max(z * 1024, 0), 1023);
            return expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
        };

        this.triangles.sort((a, b) => {
            const ax =
                (a.centroid[0] - min[0]) / bounds[0];
            const ay =
                (a.centroid[1] - min[1]) / bounds[1];
            const az =
                (a.centroid[2] - min[2]) / bounds[2];

            const bx =
                (b.centroid[0] - min[0]) / bounds[0];
            const by =
                (b.centroid[1] - min[1]) / bounds[1];
            const bz =
                (b.centroid[2] - min[2]) / bounds[2];

            return morton3D(ax, ay, az) - morton3D(bx, by, bz);
        });

        console.log("Sorting complete.");
    }

    // ------------------------------------------------------------
    // Export triangles as Float32Array
    // ------------------------------------------------------------
    getTrianglesFloat32() {
        const arr = new Float32Array(this.triangles.length * 9);
        let o = 0;

        for (const t of this.triangles) {
            arr[o++] = t.v0[0]; arr[o++] = t.v0[1]; arr[o++] = t.v0[2];
            arr[o++] = t.v1[0]; arr[o++] = t.v1[1]; arr[o++] = t.v1[2];
            arr[o++] = t.v2[0]; arr[o++] = t.v2[1]; arr[o++] = t.v2[2];
        }

        return arr;
    }

    getTriangles() {
        return this.triangles;
    }
}
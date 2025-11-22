// Scene.js
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

export class Scene {
    constructor() {
        this.loader = new GLTFLoader();
        this.triangles = [];
    }

    async loadGLB(url) {
        const gltf = await new Promise((resolve, reject) => {
            this.loader.load(
                url,
                resolve,
                (xhr) => console.log(`Loading: ${(xhr.loaded / xhr.total * 100).toFixed(1)}%`),
                (err) => {
                    console.error("GLB load failed:", err);
                    reject(err);
                }
            );
        });

        this.parseGLTF(gltf);
    }

    parseGLTF(gltf) {
        const scene = gltf.scene;

        scene.updateMatrixWorld(true); // IMPORTANT

        scene.traverse((obj) => {
            if (!obj.isMesh) return;

            const geom = obj.geometry;

            // Always convert to world-space, always flatten
            const worldMatrix = obj.matrixWorld.clone();

            const nonIndexed = geom.index !== null ? geom.toNonIndexed() : geom.clone();
            const pos = nonIndexed.getAttribute('position');

            if (!pos) return;

            const array = pos.array;
            const count = pos.count; // every 3 vertices = 1 tri

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

                this.triangles.push({
                    v0: [v0.x, v0.y, v0.z],
                    v1: [v1.x, v1.y, v1.z],
                    v2: [v2.x, v2.y, v2.z],
                });
            }
        });
    }

    getTriangles() {
        return this.triangles;
    }

    getTrianglesFloat32() {
        const arr = new Float32Array(this.triangles.length * 9);
        let ptr = 0;

        for (const t of this.triangles) {
            arr[ptr++] = t.v0[0];
            arr[ptr++] = t.v0[1];
            arr[ptr++] = t.v0[2];

            arr[ptr++] = t.v1[0];
            arr[ptr++] = t.v1[1];
            arr[ptr++] = t.v1[2];

            arr[ptr++] = t.v2[0];
            arr[ptr++] = t.v2[1];
            arr[ptr++] = t.v2[2];
        }
        return arr;
    }
}
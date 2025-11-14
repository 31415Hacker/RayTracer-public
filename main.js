import { PathTracer } from "./src/libs/PathTracer.js";
import { FPSCamera } from "./src/libs/controls/input-handler.js";
import * as THREE from "three";

window.addEventListener("DOMContentLoaded", async () => {
    /* ==========================================================
    CANVAS SETUP
    ========================================================== */

    const gpuCanvas = document.getElementById("gpuCanvas");
    const threeCanvas = document.getElementById("threeCanvas");

    // Make them full-window
    function resize() {
        gpuCanvas.width  = window.innerWidth;
        gpuCanvas.height = window.innerHeight;
        threeCanvas.width  = window.innerWidth;
        threeCanvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener("resize", resize);

    /* ==========================================================
    PATH TRACER
    ========================================================== */

    const tracer = new PathTracer(gpuCanvas);

    /* FPS Camera (JS side) */
    const FPScamera = new FPSCamera({
        canvas: gpuCanvas,
        position: [0, 0, 2.5],
        fly: true
    });

    await tracer.initialize();

    /* ==========================================================
    THREE.JS FOR BVH + CUBE WIREFRAME
    ========================================================== */

    const renderer = new THREE.WebGLRenderer({
        canvas: threeCanvas,
        alpha: true,
        antialias: true
    });
    renderer.setSize(window.innerWidth, window.innerHeight);

    // Scene + camera
    const scene = new THREE.Scene();
    const threeCam = new THREE.PerspectiveCamera(
        70,
        window.innerWidth / window.innerHeight,
        0.01,
        1000
    );

    // Group for BVH boxes
    const boxGroup = new THREE.Group();
    scene.add(boxGroup);

    /* ----------------------------------------------------------
    ADD: Cube Wireframe (the actual cube edges you expected)
    ---------------------------------------------------------- */
    {
        const cubeGeo = new THREE.BoxGeometry(2, 2, 2);
        const wireGeo = new THREE.WireframeGeometry(cubeGeo);
        const wireMat = new THREE.LineBasicMaterial({ color: 0xffffff });
        const wire = new THREE.LineSegments(wireGeo, wireMat);
        scene.add(wire);
    }

    /* ==========================================================
    BVH BOX UPDATE FUNCTION
    ========================================================== */

    function updateBVHBoxes(bvhArray) {
        boxGroup.clear();

        const numNodes = bvhArray[0];

        for (let i = 0; i < numNodes; i++) {
            const base = 1 + i * 6;
            const mn = new THREE.Vector3(
                bvhArray[base + 0],
                bvhArray[base + 1],
                bvhArray[base + 2]
            );
            const mx = new THREE.Vector3(
                bvhArray[base + 3],
                bvhArray[base + 4],
                bvhArray[base + 5]
            );

            // Make a box wireframe
            const size = new THREE.Vector3().subVectors(mx, mn);
            const center = new THREE.Vector3().addVectors(mn, mx).multiplyScalar(0.5);

            const box = new THREE.Box3(mn, mx);
            const geo = new THREE.EdgesGeometry(new THREE.BoxGeometry(size.x, size.y, size.z));
            const mat = new THREE.LineBasicMaterial({
                color: new THREE.Color(`hsl(${(i * 40) % 360},100%,60%)`)
            });
            const mesh = new THREE.LineSegments(geo, mat);

            mesh.position.copy(center);
            boxGroup.add(mesh);
        }
    }

    /* ==========================================================
    MAIN LOOP
    ========================================================== */

    let lastTime = performance.now();

    async function frame() {
        const now = performance.now();
        const dt = (now - lastTime) / 1000;
        lastTime = now;

        // Update FPS camera
        FPScamera.update(dt);

        /* ------ SYNC CAMERA INTO THE PATH TRACER ------ */
        tracer.setCameraPosition(...FPScamera.position);
        tracer.setCameraQuaternion(...FPScamera.rotation);

        /* ------ SYNC CAMERA INTO THREE.JS ------ */
        threeCam.position.set(...FPScamera.position);
        threeCam.quaternion.set(
            FPScamera.rotation[0],
            FPScamera.rotation[1],
            FPScamera.rotation[2],
            FPScamera.rotation[3]
        );

        /* ------ RAY TRACING COMPUTE PASS ------ */
        await tracer.render();

        /* ------ BVH READBACK (once per frame) ------ */
        const bvh = await tracer.readBVH();
        updateBVHBoxes(bvh);

        /* ------ WIREFRAME OVERLAY RENDER ------ */
        renderer.render(scene, threeCam);

        requestAnimationFrame(frame);
    }

    frame();
});
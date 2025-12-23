import * as PT from "./libs/PathTracer.js";
import * as PTScene from "./libs/Scene.js";
import { FPSCamera } from "./libs/controls/input-handler.js";

/* ============================================================
   DOM
============================================================ */

const FPSCounter = document.getElementById("fps");
const canvas = document.getElementById("c");

/* ============================================================
   Path Tracer + Camera
============================================================ */

const pathTracer = new PT.PathTracer(canvas);

const fpsCamera = new FPSCamera({
    canvas,
    position: [0, 0, 2.5],
    fly: true
});

await pathTracer.initialize();

/* ============================================================
   Scene
============================================================ */

const scene = new PTScene.Scene();
await scene.loadGLB("/assets/dragon.glb", {
    normalize: true,
    mode: "cube"
});
await pathTracer.setScene(scene);

/* ============================================================
   Timing / Counters
============================================================ */

// Timing
let lastFrameTime = performance.now();
let fpsTimer = performance.now();

// FPS counter (resets every second)
let fpsFrameCounter = 0;

// Monotonic frame index (never resets, goes to GPU)
let frameIndex = 0;

/* ============================================================
   Main Loop
============================================================ */

async function main() {
    /* ------------------------------
       Time
    ------------------------------ */

    const now = performance.now();
    const deltaTime = (now - lastFrameTime) / 1000;
    lastFrameTime = now;

    /* ------------------------------
       Camera update
    ------------------------------ */

    fpsCamera.update(deltaTime);

    /* ------------------------------
       FPS tracking
    ------------------------------ */

    fpsFrameCounter++;
    frameIndex++;

    if (now - fpsTimer >= 1000) {
        FPSCounter.innerText = `${fpsFrameCounter} FPS`;
        fpsFrameCounter = 0;
        fpsTimer = now;
    }

    /* ------------------------------
       Upload camera + frame index
    ------------------------------ */

    pathTracer.setCameraPosition(...fpsCamera.position);
    pathTracer.setCameraQuaternion(...fpsCamera.rotation);
    pathTracer.setFrameCount(frameIndex);

    /* ------------------------------
       Render
    ------------------------------ */

    await pathTracer.render();

    /* ------------------------------
       Next frame
    ------------------------------ */

    requestAnimationFrame(main);
}

/* ============================================================
   Start
============================================================ */

main();
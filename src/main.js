import * as PT from "./libs/PathTracer.js";
import * as PTScene from "./libs/Scene.js";
import { FPSCamera } from "./libs/controls/input-handler.js";

const FPSCounter = document.getElementById("fps");

const canvas = document.getElementById("c");
const pathTracer = new PT.PathTracer(canvas);
const FPScamera = new FPSCamera({
    canvas,
    position: [0, 0, 2.5],
    fly: true
});

await pathTracer.initialize();

const scene = new PTScene.Scene();
await scene.loadGLB('/assets/plane.glb', {
    normalize: true,
    mode: "cube"
});
await pathTracer.setScene(scene);

let lastFrame = performance.now();
let fpsTimer = performance.now();
let frameCount = 0;

async function main() {

    const now = performance.now();
    const dt = (now - lastFrame) / 1000;
    lastFrame = now;

    FPScamera.update(dt);

    // Update FPS once per second
    frameCount++;
    if (now - fpsTimer >= 1000) {
        FPSCounter.innerText = frameCount + " FPS";
        frameCount = 0;
        fpsTimer = now;
    }

    const camPos = FPScamera.position;
    const camQuat = FPScamera.rotation;

    pathTracer.setCameraPosition(...camPos);
    pathTracer.setCameraQuaternion(...camQuat);

    await pathTracer.render();

    requestAnimationFrame(main);
}

main();
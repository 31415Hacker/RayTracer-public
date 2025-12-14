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
await scene.loadGLB('/assets/dragon.glb', {
    normalize: true,
    mode: "cube"
});
await pathTracer.setScene(scene);

let lastFrame = performance.now();
let fpsTimer = performance.now();
let frameCount = 0;

let dumped = false;

async function main() {
    const now = performance.now();
    const dt = (now - lastFrame) / 1000;
    lastFrame = now;

    FPScamera.update(dt);

    frameCount++;
    if (now - fpsTimer >= 1000) {
        FPSCounter.innerText = frameCount + " FPS";
        frameCount = 0;
        fpsTimer = now;
    }

    pathTracer.setCameraPosition(...FPScamera.position);
    pathTracer.setCameraQuaternion(...FPScamera.rotation);

    await pathTracer.render();

    if (!dumped) {
        const BVH = await pathTracer.readBVH();

        const dump = BVH.slice(0, 1 + 1000 * 4);

        await fetch("http://localhost:3000/api/write", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                filename: "BVH_first_1000_nodes.json",
                content: JSON.stringify(Array.from(dump), null, 2)
            })
        });

        dumped = true;
        console.log("BVH dump written (first 1000 nodes)");
    }

    requestAnimationFrame(main);
}

main();
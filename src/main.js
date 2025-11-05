import { PathTracer } from "./libs/PathTracer.js";
import { FPSCamera } from "./libs/controls/input-handler.js";

const canvas = document.getElementById('c');
const pathTracer = new PathTracer(canvas);
const FPScamera = new FPSCamera({
    canvas,
    position: [0, 0, 2.5],
    fly: true
});

await pathTracer.initialize();

let lastTime = performance.now();

async function main() {
    const now = performance.now();
    const dt = (now - lastTime) / 1000;
    lastTime = now;

    FPScamera.update(dt);

    const camPos = FPScamera.position;
    const camQuat = FPScamera.rotation;

    pathTracer.setCameraPosition(...camPos);
    pathTracer.setCameraQuaternion(...camQuat);

    await pathTracer.render();
    requestAnimationFrame(main);
}

await main();
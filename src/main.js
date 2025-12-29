import * as PT from "./libs/PathTracer.js";
import * as PTScene from "./libs/Scene.js";
import { FPSCamera } from "./libs/controls/input-handler.js";

const canvas = document.getElementById("c");
const FPSCounter = document.getElementById("fps");

const pathTracer = new PT.PathTracer(canvas);

const fpsCamera = new FPSCamera({
  canvas,
  position: [0, 0, 2.5],
  fly: true
});

await pathTracer.initialize();

// ---------- Scene ----------
const scene = new PTScene.Scene();
await scene.loadGLB("/assets/dragon.glb", {
  normalize: true,
  mode: "cube"
});
await pathTracer.setScene(scene);

// ---------- BVH Dump (ONCE) ----------
const numTris = (pathTracer.trianglesData.length / 9) | 0;
const { bytes: bvh2Bytes } = pathTracer.computeBVH2Sizing(numTris);

const bvh2U32 = await pathTracer.readBVH2(bvh2Bytes);

console.log("Uploading BVH2:", bvh2U32.length * 4, "bytes");

let res = await fetch("http://localhost:3000/api/write", {
  method: "POST",
  headers: {
    "Content-Type": "application/octet-stream"
  },
  body: bvh2U32.buffer
});

if (!res.ok) {
  console.error("BVH2 dump failed:", await res.text());
} else {
  console.log("BVH2 dump complete");
}

// ---------- Render Loop ----------
let lastFrameTime = performance.now();
let fpsTimer = performance.now();
let fpsFrameCounter = 0;
let frameIndex = 0;

async function main() {
  const now = performance.now();
  const deltaTime = (now - lastFrameTime) / 1000;
  lastFrameTime = now;

  fpsCamera.update(deltaTime);

  fpsFrameCounter++;
  frameIndex++;

  if (now - fpsTimer >= 1000) {
    FPSCounter.innerText = `${fpsFrameCounter} FPS`;
    fpsFrameCounter = 0;
    fpsTimer = now;
  }

  pathTracer.setCameraPosition(...fpsCamera.position);
  pathTracer.setCameraQuaternion(...fpsCamera.rotation);
  pathTracer.setFrameCount(frameIndex);

  await pathTracer.render();
  requestAnimationFrame(main);
}

main();
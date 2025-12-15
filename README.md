# **GPU Path Tracer (WebGPU)**

A modern, modular **WebGPU-based path tracing engine** featuring a fully GPU-accelerated BVH builder, triangle-mesh support, and an extensible rendering pipeline. This project demonstrates how real-time ray tracing can be implemented entirely on the GPU using compute shaders.

---

## **Key Features**

### **Fully GPU-Built BVH**

* Multi-pass, breadth-first BVH construction.
* Heap-style node layout for cache-efficient traversal.
* Exact-size BVH buffer allocation based on depth.
  *(Computed using `computeBVHSizing` in `PathTracer.js`.)*

Formula for BVH buffer allocation:
```
1 + 4 * nodes
```

### **Compute Shader Ray Tracing**

* Pure compute-shader renderer (`renderer.wgsl`).
* Per-pixel ray generation, BVH traversal, and intersection logic.
* Camera controlled through position and quaternion uniforms.

### **Multi-Stage Pipeline**

The system uses three GPU pipelines:

| Stage                       | Shader            | Purpose                                         |
| --------------------------- | ----------------- | ----------------------------------------------- |
| **BVH Builder**          | `BVHBuilder.wgsl` | Builds hierarchy top-down on the GPU            |
| **Ray Tracing Renderer** | `renderer.wgsl`   | Generates rays, traverses BVH, computes shading |
| **Tonemapper**           | `tonemapper.wgsl` | Converts raw output to final display            |

### **Mesh Support**

* `three.js` is used strictly for loading/parsing GLB/GLTF files.
* Triangles stored as tightly-packed 9-float arrays.
* GPU buffer is auto-sized (up to 4 MB default).

### **Zero Precomputation Philosophy**

* No CPU-side BVH building.
* No precomputed textures for BVH or triangle data.
* Renderer queries raw GPU buffers directly.

---

## **Getting Started**

### **1. Install Dependencies**

```sh
npm install three
```

### **2. Development Mode (Hot Reload)**

Ideal for shader iteration and debugging:

```sh
npm run dev
```

### **3. Production Build**

```sh
npm run build
```

### **4. Preview Production Build**

```sh
npm run preview
```

---

## **Project Structure**

```
.
├── README.md
├── debug.html
├── index.html
├── jsconfig.json
├── main.js
├── package-lock.json
├── package.json
├── public
│   ├── app
│   │   └── styles.css
│   └── assets
│       ├── dodecahedron.glb
│       ├── plane.glb
│       └── steve.glb
├── src
│   ├── libs
│   │   ├── PathTracer.js
│   │   ├── Scene.js
│   │   ├── controls
│   │   │   └── input-handler.js
│   │   └── io.js
│   ├── main.js
│   ├── server
│   │   └── server.js
│   └── shaders
│       ├── BVHBuilder.wgsl
│       ├── renderer.wgsl
│       └── tonemapper.wgsl
└── vite.config.js
```

---

## **How It Works**

### **1. WebGPU Initialization**

`PathTracer.initialize()` requests an adapter, device, swap chain, and configures the render surface.
It then loads the WGSL shaders, allocates buffers and textures, and creates all bind groups.

### **2. BVH Building (Fully on GPU)**

Triggered via:

```js
await tracer.buildBVH(triangleData);
```

The builder:

1. Uploads triangles
2. Writes the **Builder UBO** (triangle count, max depth, batch size)
3. Dispatches compute passes
4. Finalizes BVH buffer

The BVH buffer is tightly sized using:

```js
(numNodes * 4 + 1) floats
```

as defined in the code.


### **3. Rendering**

Each frame:

* The **Renderer UBO** is updated with camera position, quaternion, resolution, triangle count, BVH node count
* Compute pass performs ray traversal
* Tonemapper draws the final full-screen triangle to the canvas

---

## **Example Usage**

```js
import PathTracer from './src/PathTracer.js';

const canvas = document.querySelector('canvas');
const tracer = new PathTracer(canvas);

await tracer.initialize();
await tracer.render();
```

Changing camera:

```js
tracer.setCameraPosition(0, 1, 3);
tracer.setCameraQuaternion(0, 0.707, 0, 0.707);
```

---

## **Design Philosophy**

* **Everything GPU-first**:
  CPU only loads files and triggers commands, and the GPU does everything else (rendering, building the BVH).
* **Extensibility**:
  Easily add materials, BRDFs, MIS, ReSTIR, or sampling strategies.
* **Separation of concerns**:

  * `PathTracer.js` handles GPU orchestration.
  * WGSL handles math, geometry, sampling, and traversal.

---

## **Roadmap**

* [ ] Add reflections and refractions
* [ ] Add multi-bounce GI
* [ ] Add Mesh transforms
* [x] Add BVH debugging visualizer
* [ ] Add OBJ/PLY support
* [ ] Add full scene materials + textures

---

## **License**

MIT — free to use, modify, and learn from.
# **A Realtime WebGPU PathTracer**
This is a **path tracer**, specifically made for running the expensive but beautiful algorithm on low-end hardware. It is very structured, and has the following packages:
- Vite
- Node
- THREE.js

> In summary, the path tracer aims for a low-latency, high-quality renderer that can be used for a variety of applications, including 3D rendering, physics simulations, and more using low-end hardware.

## *Using* the **Path Tracer**
Before we install and use the path tracer, we need some prerequisites.

### **Prerequisites**
1. **Node.js** - This is the runtime environment for running the code. You can download it from [nodejs.org](https://nodejs.org/en/), or use a package manager.
Here's an example in **Linux Ubuntu 22.04 LTS**:
```bash
sudo apt install node
```
We need the latest version, so it is recommended to use something like **Git** to install it.

2. **Vite** - This is a build tool that helps us build our code into a single file, which is then translated into a browser-friendly format. You can install it using npm:

```bash
npm install -g vite
```
We also need the latest version of Vite, so it is recommended to use something like **Git** to install it alongside Node & NPM as well.

3. **WebGPU**

WebGPU is a new API for rendering graphics on the GPU. It allows you to render graphics on the GPU, which is faster than rendering on the CPU. It is not a package, but you can enable it in a web browser by enabling the flags. Search for "webgpu" in the browser's flags (`[BROWSER NAME]://flags`) to enable it. Examples of browsers who support WebGPU are:
- Chrome
- Firefox
- Edge
- Opera
- Vivaldi

> Some other browsers might not support WebGPU, so be aware of this limitation before using the Path Tracer.

4. **THREE.JS**
Three js is essential but not completely neccesary if you really want to ditch it. It is primarily used for parsing the mesh data, so you can replace this with a custom parser if you like. Here's how to install it with npm:
```bash
npm install three
```

---
We're finally done with the prerequisites! Let's move on to **Step 2**, actually downloading the project itself!

### **Downloading the Repository**
This is pretty straightforward, all you need to do is install git (Linux Ubuntu 22.04 LTS):
```bash
sudo apt install git
```

Pull the repository:
```bash
git init
git clone https://github.com/31415Hacker/RayTracer # [Whatever Folder You Like]
```

And... that's it!

---

After prerequisites and downloading the project, we get to the fun stuff - *using* the path tracer!

### Using the Path Tracer

Now, the fun stuff! Here are some tips for using the path tracer, as of now of the 12-25-2025 version:

- It is a web app, so you need to run two commands:
```bash
npm run dev
```

> Initializes the web app itself.

and

```bash
npm run api
```

> Initializes the node.js server for API calls.

After running the web app, you just need to navigate to the link that the first command gave you, and by default it is this link: [WebGPU Website](http://localhost:5173/).

Inside the app, you can use **WASD** for moving forward, left, right, and backward while using **Q & E** go up and down. To rotate, it's just a pointer lock so you know what to do. There are usually no other things that can be changed by the UI *alone*, so you have to edit the code or files to:
- Change Models (go to `public/assets/`, put a model, GLB to be exact and in `src/main.js`, and tweak the `loadGLB` to contain your mesh instead of the default one. You also might need to tweak BVH settings like `maxDepth` to work properly.)
- Change Lighting (You'll need to go to the shader, `src/shaders/renderer.wgsl`)
- And Change Anything Else

---

That's it for now! I will include extra information inside this README in the future.
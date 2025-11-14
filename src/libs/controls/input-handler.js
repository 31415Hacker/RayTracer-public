// /src/libs/controls/input-handler.js
// FPS camera controller with selectable movement frame:
//  - fly=true  : full camera-space (WASD follows yaw+pitch; Space/Ctrl = camera up/down)
//  - fly=false : yaw-only ground walker (WASD projected to XZ; Space/Ctrl = world up/down)
import { vec3, quat, mat4 } from "gl-matrix";

export const CAM_VERSION = 4;

const X_AXIS = [1, 0, 0];
const Y_AXIS = [0, 1, 0];
const PITCH_CLAMP = Math.PI / 2 - 0.0001;

// local basis vectors (camera space)
const FWD_L = vec3.fromValues(0, 0, -1);
const RIGHT_L = vec3.fromValues(1, 0, 0);
const UP_L = vec3.fromValues(0, 1, 0);

export class FPSCamera {
  // pose
  #pos = vec3.fromValues(0, 1.6, 5);
  #yaw = 0;
  #pitch = 0;
  #q = quat.create(); // [x,y,z,w]

  // config
  #base = 4.0;
  #sprint = 2.5;
  #sens = 0.002;
  #fly = true; // << set true for camera-space, false for ground (yaw-only)

  #canvas;
  #keys = new Set();
  #plock = false;

  // scratch to avoid GC in update()
  #tmpFwd = vec3.create();
  #tmpRight = vec3.create();
  #tmpUp = vec3.create();
  #move = vec3.create();

  constructor({
    canvas,
    position = [0, 1.6, 5],
    moveSpeed = 4.0,
    sprintMult = 2.5,
    lookSensitivity = 0.002,
    fly = true, // NEW: choose camera-space vs ground-walk
  } = {}) {
    if (!canvas) throw new Error("FPSCamera needs { canvas }");
    this.#canvas = canvas;
    vec3.copy(this.#pos, position);
    this.#base = moveSpeed;
    this.#sprint = sprintMult;
    this.#sens = lookSensitivity;
    this.#fly = fly;

    // pointer lock + input
    document.addEventListener("pointerlockchange", () => {
      this.#plock = document.pointerLockElement === this.#canvas;
    });
    this.#canvas.addEventListener("click", () => {
      if (!this.#plock) this.#canvas.requestPointerLock?.();
    });
    document.addEventListener("mousemove", (e) => {
      if (!this.#plock) return;
      this.#yaw -= e.movementX * this.#sens;
      this.#pitch -= e.movementY * this.#sens;
      this.#pitch = clamp(this.#pitch, -PITCH_CLAMP, PITCH_CLAMP);
    });

    // prevent page scroll on Space/Arrows while controlling
    window.addEventListener("keydown", (e) => {
      if (
        this.#plock &&
        (e.code === "Space" ||
          e.code === "ArrowUp" ||
          e.code === "ArrowDown" ||
          e.code === "ArrowLeft" ||
          e.code === "ArrowRight")
      ) {
        e.preventDefault();
      }
      this.#keys.add(e.code);
    });
    window.addEventListener("keyup", (e) => this.#keys.delete(e.code));
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) this.#keys.clear();
    });
  }

  // Toggle movement frame at runtime if desired
  setFly(enabled) {
    this.#fly = !!enabled;
  }
  get fly() {
    return this.#fly;
  }

  update(dt) {
    // --- Build render orientation: q = yaw(world-Y) * pitch(local-X) ---
    const qYaw = quat.setAxisAngle(quat.create(), Y_AXIS, this.#yaw);
    const qPitch = quat.setAxisAngle(quat.create(), X_AXIS, this.#pitch);
    quat.multiply(this.#q, qYaw, qPitch);
    quat.normalize(this.#q, this.#q);

    // --- Build camera-space basis from quaternion ---
    // fwd/right/up are camera axes in world space
    vec3.transformQuat(this.#tmpFwd, FWD_L, this.#q);
    vec3.transformQuat(this.#tmpRight, RIGHT_L, this.#q);
    vec3.transformQuat(this.#tmpUp, UP_L, this.#q);

    // If not flying, constrain movement to ground plane (yaw-only)
    if (!this.#fly) {
      this.#tmpFwd[1] = 0;
      this.#tmpRight[1] = 0;
      // Renormalize after projection to keep speed consistent
      const lf = vec3.length(this.#tmpFwd);
      const lr = vec3.length(this.#tmpRight);
      if (lf > 1e-6) vec3.scale(this.#tmpFwd, this.#tmpFwd, 1 / lf);
      if (lr > 1e-6) vec3.scale(this.#tmpRight, this.#tmpRight, 1 / lr);
      // Up/down remain world-up for ground mode
      vec3.set(this.#tmpUp, 0, 1, 0);
    }

    // --- Accumulate input in chosen movement frame ---
    vec3.set(this.#move, 0, 0, 0);
    if (this.#keys.has("KeyW")) vec3.add(this.#move, this.#move, this.#tmpFwd);
    if (this.#keys.has("KeyS")) vec3.sub(this.#move, this.#move, this.#tmpFwd);
    if (this.#keys.has("KeyD")) vec3.add(this.#move, this.#move, this.#tmpRight);
    if (this.#keys.has("KeyA")) vec3.sub(this.#move, this.#move, this.#tmpRight);
    if (this.#keys.has("KeyE")) vec3.add(this.#move, this.#move, this.#tmpUp);
    if (this.#keys.has("KeyQ")) vec3.sub(this.#move, this.#move, this.#tmpUp);

    const len = vec3.length(this.#move);
    if (len > 1e-6) {
      vec3.scale(this.#move, this.#move, 1 / len);
      const speed =
        this.#keys.has("ShiftLeft") || this.#keys.has("ShiftRight")
          ? this.#base * this.#sprint
          : this.#base;
      vec3.scaleAndAdd(this.#pos, this.#pos, this.#move, speed * dt);
    }
  }

  // --- UBO accessors ---
  get position() {
    return this.#pos;
  }
  get rotation() {
    return this.#q;
  } // [x,y,z,w]

  // Optional helper if you want a packed array:
  toArray() {
    return [
      this.#pos[0],
      this.#pos[1],
      this.#pos[2],
      0,
      this.#q[0],
      this.#q[1],
      this.#q[2],
      this.#q[3],
    ];
  }
}

// utils
function clamp(x, a, b) {
  return x < a ? a : x > b ? b : x;
}

export default FPSCamera;
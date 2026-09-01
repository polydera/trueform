import "three/addons/controls/ArcballControls.js";
import type { Vector3 } from "three";

/**
 * ArcballControls carries these at runtime (three/examples/jsm), but
 * @types/three does not declare them. The docs' scenes read all three:
 * the orbit target, the gizmo toggle, and the state restore that drives
 * the synced side-by-side viewers.
 */
declare module "three/addons/controls/ArcballControls.js" {
  interface ArcballControls {
    target: Vector3;
    enableGizmos: boolean;
    setStateFromJSON(json: string): void;
  }
}

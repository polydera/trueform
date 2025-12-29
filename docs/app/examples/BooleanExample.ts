import type { MainModule } from "@/examples/native";
import { syncOrbitControls, fitCameraToAllMeshesFromZPlane } from "@/utils/sceneUtils";
import {
  buffersToCurves,
  createMesh,
  updateResultMesh,
  CurveRenderer,
} from "@/utils/utils";
import { ThreejsBase } from "@/examples/ThreejsBase";
import * as THREE from "three";

export class BooleanExample extends ThreejsBase {
  private curveRenderer: CurveRenderer;
  private keyPressed = false;

  // private pointDebug = createPoints();
  public randomize() {
    this.wasmInstance.OnKeyPress("n");
    this.updateMeshes();
  }

  public resyncCamera() {
    this.syncSceneControls = true;
    if (this.sceneBundle2) {
      syncOrbitControls(this.sceneBundle1.controls, this.sceneBundle2.controls);
    }
  }

  constructor(
    wasmInstance: MainModule,
    path: string[],
    container: HTMLElement,
    container2: HTMLElement,
    isDarkMode = true,
  ) {
    super(wasmInstance, path, container, container2, true, false, isDarkMode);

    // Enable smooth shading for the sphere (meshDataId = 1)
    const sphereGeometry = this.geometries.get(1);
    const sphereInstancedMesh = this.instancedMeshes.get(1);
    if (sphereGeometry && sphereInstancedMesh) {
      sphereGeometry.computeVertexNormals();
      const oldMaterial = sphereInstancedMesh.material as THREE.MeshMatcapMaterial;
      const smoothMaterial = new THREE.MeshMatcapMaterial({
        matcap: oldMaterial.matcap,
        side: THREE.DoubleSide,
        flatShading: false,
      });
      sphereInstancedMesh.material = smoothMaterial;
    }

    const interceptKeyDownEvent = (event: KeyboardEvent) => {
      if (this.keyPressed) return;
      this.keyPressed = true;
      if (event.key === "r") {
        this.resyncCamera();
        return;
      }
      if (event.key === "n") {
        this.randomize();
        return;
      }
      this.wasmInstance.OnKeyPress(event.key);
      this.updateMeshes();
    };
    const interceptKeyUpEvent = (_event: KeyboardEvent) => {
      this.keyPressed = false;
    };
    window.addEventListener("keydown", interceptKeyDownEvent);
    window.addEventListener("keyup", interceptKeyUpEvent);
    this.addCleanup(() => {
      window.removeEventListener("keydown", interceptKeyDownEvent);
      window.removeEventListener("keyup", interceptKeyUpEvent);
    });

    // Ctrl+scroll to change sphere radius
    const interceptWheelEvent = (event: WheelEvent) => {
      if (!event.ctrlKey) return;
      event.preventDefault();
      const delta = event.deltaY !== 0 ? event.deltaY : event.deltaX;
      if (delta === 0) return;
      const normalizedDelta = delta / Math.abs(delta);
      const handled = this.wasmInstance.OnMouseWheel(-normalizedDelta, true);
      this.updateMeshes();
      if (handled) {
        event.stopImmediatePropagation();
      }
    };
    const wheelListenerOptions = {
      passive: false,
      capture: true,
    };
    window.addEventListener("wheel", interceptWheelEvent, wheelListenerOptions);
    this.addCleanup(() => {
      window.removeEventListener("wheel", interceptWheelEvent, wheelListenerOptions);
    });

    this.curveRenderer = new CurveRenderer({
      color: 0xff2020,
      radius: 0.075,
      maxSegments: 20000,
    });
    this.sceneBundle1.scene.add(this.curveRenderer.object);

    if (this.sceneBundle2 && this.renderer2) {
      const mesh = createMesh(this.isDarkMode);
      this.meshes2.set(0, mesh);
      this.sceneBundle2.scene.add(mesh);
    }

    this.updateMeshes();
    fitCameraToAllMeshesFromZPlane(this.sceneBundle1, 1.8);

    // Adjust camera angle slightly right and up for better dragon view
    const camera = this.sceneBundle1.camera;
    const target = this.sceneBundle1.controls.target;
    const offset = camera.position.clone().sub(target);
    const distance = offset.length();
    const angleH = 0.15; // horizontal angle, positive = right
    const angleV = 0.1; // vertical angle, positive = up
    camera.position.set(
      target.x + distance * Math.sin(angleH),
      target.y + distance * Math.sin(angleV),
      target.z + distance * Math.cos(angleH) * Math.cos(angleV)
    );
    camera.lookAt(target);
    this.sceneBundle1.controls.update();

    if (this.sceneBundle2) {
      fitCameraToAllMeshesFromZPlane(this.sceneBundle2, 1.8);
      syncOrbitControls(this.sceneBundle1.controls, this.sceneBundle2.controls);
    }
  }

  public runMain() {
    const v = new this.wasmInstance.VectorString();
    for (let i = 0; i < this.paths.length; i++) {
      v.push_back(this.paths[i]!);
    }
    this.wasmInstance.run_main(v);
    for (let i = 0; i < this.paths.length; i++) {
      this.wasmInstance.FS.unlink(this.paths[i]);
    }
  }

  public override updateMeshes() {
    super.updateMeshes();

    // Update curve mesh (intersection curves)
    const cO = this.wasmInstance.get_curve_mesh();
    if (cO && cO.updated) {
      const points = cO.get_curve_points();
      const ids = cO.get_curve_ids();
      const offsets = cO.get_curve_offsets();
      const curves = buffersToCurves(points, ids, offsets);
      this.curveRenderer.update(curves);
    }

    if (this.renderer2 && this.sceneBundle2) {
      const resultMesh = this.wasmInstance.get_result_mesh();
      const mesh = this.meshes2.get(0);
      if (resultMesh && mesh) {
        updateResultMesh(resultMesh, mesh);
      }
    }
  }
}

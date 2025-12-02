import type { MainModule } from "@/examples/native";
import { fitCameraToAllMeshesFromZPlane, syncOrbitControls } from "@/utils/sceneUtils";
import {
  buffersToCurves,
  createMesh,
  getMeshFromWasm,
  CurveRenderer,
} from "@/utils/utils";
import { ThreejsBase } from "@/examples/ThreejsBase";

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
    if (!this.syncSceneControls && this.sceneBundle2) {
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
    if (cO && cO.polydata_updated) {
      const points = cO.get_curve_points();
      const ids = cO.get_curve_ids();
      const offsets = cO.get_curve_offsets();
      const curves = buffersToCurves(points, ids, offsets);
      this.curveRenderer.update(curves);
    }

    if (this.renderer2 && this.sceneBundle2) {
      const wO = this.wasmInstance.get_result_mesh();
      const mesh = this.meshes2.get(0);
      if (wO && mesh) {
        getMeshFromWasm(wO, mesh);
      }
    }
    // const t1 = performance.now();
    // console.log("updateMeshes took " + (t1 - t0) + "ms");
  }
}

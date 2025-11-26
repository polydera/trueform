import type { MainModule } from "@/examples/native";
import { fitCameraToAllMeshesFromZPlane } from "@/utils/sceneUtils";
import {
  buffersToCurves,
  createCurveLineObjects,
  createMesh,
  type CurveLineObjects,
  curvesToCurveLines,
  type curvesToCurvePolyOpts,
  getMeshFromWasm,
  createBasicCurveLineObjects,
  updateBasicCurveLines,
} from "@/utils/utils";
import { ThreejsBase } from "@/examples/ThreejsBase";

export class BooleanExample extends ThreejsBase {
  private curveObjects: CurveLineObjects | any;
  private useBasicLines = false; // Set to true to use basic LineSegments for debugging
  private keyPressed = false;

  // private pointDebug = createPoints();

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

    const opts: curvesToCurvePolyOpts = {
      tubeColor: 0xff2020, // Red lines
      lineWidth: 0.2,
    };

    if (this.useBasicLines) {
      // Use basic LineSegments for compatibility
      this.curveObjects = createBasicCurveLineObjects(opts);
      console.log("Using basic LineSegments for curve rendering");
    } else {
      this.curveObjects = createCurveLineObjects(opts);
      console.log("Using LineSegments2 for curve rendering");
    }
    this.sceneBundle1.scene.add(this.curveObjects.lines);

    if (this.sceneBundle2 && this.renderer2) {
      const mesh = createMesh(this.isDarkMode);
      this.meshes2.set(0, mesh);
      this.sceneBundle2.scene.add(mesh);
    }

    this.updateMeshes();
    fitCameraToAllMeshesFromZPlane(this.sceneBundle1, 1.5);
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
    // const t0 = performance.now();
    super.updateMeshes();

    // Update curve mesh (intersection curves)
    const cO = this.wasmInstance.get_curve_mesh();
    if (cO && cO.polydata_updated) {
      const points = cO.get_curve_points();
      const ids = cO.get_curve_ids();
      const offsets = cO.get_curve_offsets();
      const lines = buffersToCurves(points, ids, offsets);
      if (this.useBasicLines) {
        updateBasicCurveLines(lines, this.curveObjects);
      } else {
        curvesToCurveLines(lines, this.curveObjects, { samplesPerSegment: 3, tension: 0.5 });
      }
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

import type { MainModule } from "@/examples/native";
import { fitCameraToAllMeshesFromZPlane } from "@/utils/sceneUtils";
import {
  buffersToCurves,
  createBasicCurveLineObjects,
  createCurveLineObjects,
  type CurveLineObjects,
  curvesToCurveLines,
  type curvesToCurvePolyOpts,
  updateBasicCurveLines,
} from "@/utils/utils";
import { ThreejsBase } from "@/examples/ThreejsBase";

export class FormsIntersectionsExample extends ThreejsBase {
  private curveObjects: CurveLineObjects | any;
  private useBasicLines = false;
  private keyPressed = false;
  public randomize() {
    this.wasmInstance.OnKeyPress("n");
    this.updateMeshes();
  }

  constructor(
    wasmInstance: MainModule,
    paths: string[],
    container: HTMLElement,
    isDarkMode = true,
  ) {
    super(wasmInstance, paths, container, undefined, true, false, isDarkMode);

    const interceptKeyDownEvent = (event: KeyboardEvent) => {
      if (this.keyPressed) return;
      this.keyPressed = true;
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

    const opts: curvesToCurvePolyOpts = {
      tubeColor: 0xff2020,
      lineWidth: 0.2,
    };

    this.curveObjects = this.useBasicLines
      ? createBasicCurveLineObjects(opts)
      : createCurveLineObjects(opts);
    this.sceneBundle1.scene.add(this.curveObjects.lines);

    this.updateMeshes();
    fitCameraToAllMeshesFromZPlane(this.sceneBundle1);
  }

  public runMain() {
    const v = new this.wasmInstance.VectorString();
    for (const path of this.paths) {
      v.push_back(path);
    }
    this.wasmInstance.run_main_forms_intersections(v);
    for (const path of this.paths) {
      this.wasmInstance.FS.unlink(path);
    }
  }

  public override updateMeshes() {
    super.updateMeshes();

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
  }
}

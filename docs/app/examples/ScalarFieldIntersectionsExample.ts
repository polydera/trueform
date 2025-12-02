import type { MainModule } from "@/examples/native";
import { fitCameraToAllMeshesFromZPlane } from "@/utils/sceneUtils";
import {
  buffersToCurves,
  createBasicCurveLineObjects,
  createCurveLineObjects,
  curvesToCurveLines,
  type curvesToCurvePolyOpts,
  updateBasicCurveLines,
} from "@/utils/utils";
import { ThreejsBase } from "@/examples/ThreejsBase";

export class ScalarFieldIntersectionsExample extends ThreejsBase {
  private curveObjects: any;
  private useBasicLines = false;
  private keyPressed = false;

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
      this.wasmInstance.OnKeyPress(event.key);
      this.updateMeshes();
    };
    const interceptKeyUpEvent = (_event: KeyboardEvent) => {
      this.keyPressed = false;
    };
    window.addEventListener("keydown", interceptKeyDownEvent);
    window.addEventListener("keyup", interceptKeyUpEvent);

    const interceptWheelEvent = (event: WheelEvent) => {
      if (!event.shiftKey) return;
      event.preventDefault();
      const delta = event.deltaY !== 0 ? event.deltaY : event.deltaX;
      if (delta === 0) return;
      const normalizedDelta = delta / Math.abs(delta);
      const handled = this.wasmInstance.OnMouseWheel(normalizedDelta, event.shiftKey);
      this.updateMeshes();
      if(handled)
        event.stopImmediatePropagation();
    };
    const wheelListenerOptions = {
      passive: false,
      capture: true
    };
    window.addEventListener("wheel", interceptWheelEvent, wheelListenerOptions);
    this.addCleanup(() => {
      window.removeEventListener("keydown", interceptKeyDownEvent);
      window.removeEventListener("keyup", interceptKeyUpEvent);
      window.removeEventListener("wheel", interceptWheelEvent, wheelListenerOptions);
    });

    const opts: curvesToCurvePolyOpts = {
      tubeColor: 0xffaa00,
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
    this.wasmInstance.run_main_scalar_field_intersections(this.paths[0]!);
    this.wasmInstance.FS.unlink(this.paths[0]);
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

import type { MainModule } from "@/examples/native";
import { fitCameraToAllMeshesFromZPlane } from "@/utils/sceneUtils";
import {
  buffersToCurves,
  createBasicCurveLineObjects,
  createCurveLineObjects,
  curvesToCurveLines,
  type curvesToCurvePolyOpts,
  createMesh,
  getMeshFromWasm,
  updateBasicCurveLines,
} from "@/utils/utils";
import { ThreejsBase } from "@/examples/ThreejsBase";

export class IsobandsExample extends ThreejsBase {
  private curveObjects: any;
  private useBasicLines = false;
  private keyPressed = false;

  constructor(
    wasmInstance: MainModule,
    paths: string[],
    container: HTMLElement,
    container2: HTMLElement,
    isDarkMode = true,
  ) {
    super(wasmInstance, paths, container, container2, true, false, isDarkMode);

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
    window.addEventListener("wheel", interceptWheelEvent, {
      passive: false,
      capture: true
    });
    this.addCleanup(() => {
      window.removeEventListener("keydown", interceptKeyDownEvent);
      window.removeEventListener("keyup", interceptKeyUpEvent);
      window.removeEventListener("wheel", interceptWheelEvent);
    });

    const opts: curvesToCurvePolyOpts = {
      tubeColor: 0x2020ff,
      lineWidth: 0.2,
    };
    this.curveObjects = this.useBasicLines
      ? createBasicCurveLineObjects(opts)
      : createCurveLineObjects(opts);
    this.sceneBundle1.scene.add(this.curveObjects.lines);

    if (this.sceneBundle2 && this.renderer2) {
      const mesh = createMesh(this.isDarkMode);
      this.meshes2.set(0, mesh);
      this.sceneBundle2.scene.add(mesh);
    }

    this.updateMeshes();
    fitCameraToAllMeshesFromZPlane(this.sceneBundle1);
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

    if (this.renderer2 && this.sceneBundle2) {
      const wO = this.wasmInstance.get_result_mesh();
      const mesh = this.meshes2.get(0);
      if (wO && mesh) {
        getMeshFromWasm(wO, mesh);
      }
    }
  }

  public runMain() {
    this.wasmInstance.run_main_isobands(this.paths[0]!);
    this.wasmInstance.FS.unlink(this.paths[0]);
  }
}

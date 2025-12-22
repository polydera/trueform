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
    this.sceneBundle1.controls.enableZoom = false;

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

    const isEventFromCanvas = (eventTarget: EventTarget | null) => {
      if (!eventTarget) return false;
      const target = eventTarget as Node;
      const container = this.renderer.domElement.parentElement;
      return !!container && container.contains(target);
    };

    const interceptWheelEvent = (event: WheelEvent) => {
      if (!isEventFromCanvas(event.target)) return;
      const delta = event.deltaY !== 0 ? event.deltaY : event.deltaX;
      if (delta === 0) return;
      event.preventDefault();
      const normalizedDelta = delta / Math.abs(delta);
      const handled = this.wasmInstance.OnMouseWheel(normalizedDelta, true);
      this.updateMeshes();
      if(handled)
        event.stopImmediatePropagation();
    };
    const wheelListenerOptions = {
      passive: false,
      capture: true
    };
    window.addEventListener("wheel", interceptWheelEvent, wheelListenerOptions);

    let touchScrollActive = false;
    let lastTouchY = 0;
    const setTouchScrollMode = (active: boolean) => {
      touchScrollActive = active;
      const controlsEnabled = !active;
      this.sceneBundle1.controls.enabled = controlsEnabled;
      if (this.sceneBundle2) {
        this.sceneBundle2.controls.enabled = controlsEnabled;
      }
    };
    const touchScrollThresholdPx = 10;
    const getAverageTouchY = (touches: TouchList) => {
      let sum = 0;
      for (let i = 0; i < touches.length; i++) {
        sum += touches[i]!.clientY;
      }
      return sum / touches.length;
    };

    const interceptTouchStart = (event: TouchEvent) => {
      if (event.touches.length !== 1) return;
      if (!isEventFromCanvas(event.target)) return;
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      setTouchScrollMode(true);
      lastTouchY = getAverageTouchY(event.touches);
    };

    const interceptTouchMove = (event: TouchEvent) => {
      if (!touchScrollActive) return;
      if (event.touches.length !== 1) {
        setTouchScrollMode(false);
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      const currentY = getAverageTouchY(event.touches);
      const deltaY = currentY - lastTouchY;
      if (Math.abs(deltaY) < touchScrollThresholdPx) return;
      const normalizedDelta = deltaY / Math.abs(deltaY);
      const handled = this.wasmInstance.OnMouseWheel(normalizedDelta, true);
      this.updateMeshes();
      if (handled) {
        event.stopImmediatePropagation();
      }
      lastTouchY = currentY;
    };

    const interceptTouchEnd = (event: TouchEvent) => {
      if (touchScrollActive) {
        setTouchScrollMode(false);
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
      }
    };

    const touchListenerOptions = {
      passive: false,
      capture: true,
    };
    window.addEventListener("touchstart", interceptTouchStart, touchListenerOptions);
    window.addEventListener("touchmove", interceptTouchMove, touchListenerOptions);
    window.addEventListener("touchend", interceptTouchEnd, touchListenerOptions);
    window.addEventListener("touchcancel", interceptTouchEnd, touchListenerOptions);

    this.addCleanup(() => {
      window.removeEventListener("keydown", interceptKeyDownEvent);
      window.removeEventListener("keyup", interceptKeyUpEvent);
      window.removeEventListener("wheel", interceptWheelEvent, wheelListenerOptions);
      window.removeEventListener("touchstart", interceptTouchStart, touchListenerOptions);
      window.removeEventListener("touchmove", interceptTouchMove, touchListenerOptions);
      window.removeEventListener("touchend", interceptTouchEnd, touchListenerOptions);
      window.removeEventListener("touchcancel", interceptTouchEnd, touchListenerOptions);
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
    if (cO && cO.updated) {
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

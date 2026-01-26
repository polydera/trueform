import type { MainModule } from "@/examples/native";
import {fitCameraToAllMeshesFromZPlane, syncOrbitControls} from "@/utils/sceneUtils";
import {
  buffersToCurves,
  createMesh,
  updateResultMesh,
  CurveRenderer,
} from "@/utils/utils";
import { ThreejsBase } from "@/examples/ThreejsBase";

export class IsobandsExample extends ThreejsBase {
  private curveRenderer: CurveRenderer;
  private keyPressed = false;
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
    paths: string[],
    container: HTMLElement,
    container2: HTMLElement,
    isDarkMode = true,
  ) {
    super(wasmInstance, paths, container, container2, true, false, isDarkMode);
    this.sceneBundle1.controls.enableZoom = false;
    if (this.sceneBundle2) {
      this.sceneBundle2.controls.enableZoom = false;
    }

    const interceptKeyDownEvent = (event: KeyboardEvent) => {
      if (this.keyPressed) return;
      this.keyPressed = true;
      if(event.key === "r"){
        this.resyncCamera();
        return;
      }
      if(event.key === "n"){
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
      const container1 = this.renderer.domElement.parentElement;
      if (container1 && container1.contains(target)) return true;
      const container2 = this.renderer2?.domElement.parentElement;
      return !!container2 && container2.contains(target);
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

    this.curveRenderer = new CurveRenderer({
      color: 0x00d5be,
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
    fitCameraToAllMeshesFromZPlane(this.sceneBundle1, 1.5);
    if(this.sceneBundle2){
      fitCameraToAllMeshesFromZPlane(this.sceneBundle2, 1.5);
      syncOrbitControls(this.sceneBundle1.controls, this.sceneBundle2.controls);
    }
  }

  public override updateMeshes() {
    super.updateMeshes();

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

  public runMain() {
    this.wasmInstance.run_main_isobands(this.paths[0]!);
    this.wasmInstance.FS.unlink(this.paths[0]);
  }
}

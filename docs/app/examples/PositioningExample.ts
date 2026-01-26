import type { MainModule } from "@/examples/native";
import { fitCameraToAllMeshesFromZPlane } from "@/utils/sceneUtils";
import { ThreejsBase } from "@/examples/ThreejsBase";
import * as THREE from "three";

export class PositioningExample extends ThreejsBase {
  private keyPressed = false;
  constructor(
    wasmInstance: MainModule,
    paths: string[],
    container: HTMLElement,
    isDarkMode = true,
  ) {
    super(wasmInstance, paths, container, undefined, false, false, isDarkMode);

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

    fitCameraToAllMeshesFromZPlane(this.sceneBundle1, 1.5);
  }

  public randomize() {
    this.wasmInstance.OnKeyPress("n");
    this.updateMeshes();
  }

  public override onPointerUp(event: PointerEvent) {
    const cameraPosition = this.sceneBundle1.camera.position.clone();
    const dir = new THREE.Vector3();
    this.sceneBundle1.camera.getWorldDirection(dir);
    const cameraFocalPoint = cameraPosition.clone().add(dir.multiplyScalar(100));

    const updateFocalPoint = (x: number, y: number, z: number) => {
      this.sceneBundle1.controls.target.set(x, y, z);
      this.sceneBundle1.controls.update();

      // Update instanced mesh matrices
      const tempMatrix = new THREE.Matrix4();
      const needsUpdate = new Set<number>();

      for (let i = 0; i < this.wasmInstance.get_number_of_instances(); i++) {
        const inst = this.wasmInstance.get_instance_on_idx(i);
        const indices = this.instanceIndices.get(i);
        if (!inst || !indices) continue;

        const { meshDataId, indexInBatch } = indices;
        const instancedMesh = this.instancedMeshes.get(meshDataId);
        if (!instancedMesh) continue;

        const matrix = new Float32Array(inst.get_matrix());
        tempMatrix.fromArray(matrix);
        tempMatrix.transpose();
        instancedMesh.setMatrixAt(indexInBatch, tempMatrix);
        needsUpdate.add(meshDataId);
      }

      for (const meshDataId of needsUpdate) {
        const mesh = this.instancedMeshes.get(meshDataId);
        if (mesh) mesh.instanceMatrix.needsUpdate = true;
      }

      this.sceneBundle1.scene.updateMatrixWorld(true);
      this.renderer.render(this.sceneBundle1.scene, this.sceneBundle1.camera);
    };

    let t = 0;
    const stepPositioning = () => {
      if (this.isDisposed()) return;
      t = this.wasmInstance.OnLeftButtonUpCustom(
        [cameraFocalPoint.x, cameraFocalPoint.y, cameraFocalPoint.z],
        updateFocalPoint,
        t,
      );
      if (t < 1.0) {
        requestAnimationFrame(stepPositioning);
      }
      this.updateMeshes();
      if (t < 1) {
        event.stopPropagation();
      }
    };
    requestAnimationFrame(stepPositioning);
  }

  public runMain() {
    const v = new this.wasmInstance.VectorString();
    for (const path of this.paths) {
      v.push_back(path);
    }
    this.wasmInstance.run_main_positioning(v);
    for (const path of this.paths) {
      this.wasmInstance.FS.unlink(path);
    }
  }
}

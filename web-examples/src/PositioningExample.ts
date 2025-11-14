import { MainModule } from './webAssembly/dist/native.js';
import { fitCameraToAllMeshesFromZPlane } from './utils/sceneUtils';
import { TestClassThreejsBase } from "@/TestThreejsBase";
import * as THREE from "three";
import {n} from "vite/dist/node/chunks/moduleRunnerTransport";

export class PositioningExample extends TestClassThreejsBase {
    constructor(wasmInstance: MainModule, paths: string[], container: HTMLElement) {
        super(wasmInstance, paths, container);

        fitCameraToAllMeshesFromZPlane(this.sceneBundle1, 1.5);
    }

    public onPointerUp(event: PointerEvent) {
        console.log("PositioningExample onPointerUp", event.buttons)
        const cameraPosition = this.sceneBundle1.camera.position.clone();
        const dir = new THREE.Vector3();
        this.sceneBundle1.camera.getWorldDirection(dir);
        const cameraFocalPoint = cameraPosition.clone().add(dir.multiplyScalar(100));
        const update_focal_point_lambda = (x: number, y: number, z: number) => {
            console.log("PositioningExample update_focal_point_lambda", x, y, z)
            this.sceneBundle1.controls.target.set(x, y, z);
            this.sceneBundle1.controls.update();
            this.updateMeshes();
            this.sceneBundle1.scene.updateMatrixWorld(true);
            this.renderer.render(this.sceneBundle1.scene, this.sceneBundle1.camera);
        }
        const handled = this.wasmInstance.OnLeftButtonUpCustom([cameraFocalPoint.x, cameraFocalPoint.y, cameraFocalPoint.z], update_focal_point_lambda);
        this.updateMeshes()
        if (handled) {
            event.stopPropagation();
        }
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

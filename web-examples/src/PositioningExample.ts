import { MainModule } from './webAssembly/dist/native.js';
import { fitCameraToAllMeshesFromZPlane } from './utils/sceneUtils';
import { TestClassThreejsBase } from "@/TestThreejsBase";
import * as THREE from "three";

export class PositioningExample extends TestClassThreejsBase {
    constructor(wasmInstance: MainModule, paths: string[], container: HTMLElement) {
        super(wasmInstance, paths, container);

        fitCameraToAllMeshesFromZPlane(this.sceneBundle1, 1.5);
    }

    public onPointerUp(event: PointerEvent) {
        const cameraPosition = this.sceneBundle1.camera.position.clone();
        const dir = new THREE.Vector3();
        this.sceneBundle1.camera.getWorldDirection(dir);
        const cameraFocalPoint = cameraPosition.clone().add(dir.multiplyScalar(100));

        const update_focal_point_lambda = (x: number, y: number, z: number) => {
            this.sceneBundle1.controls.target.set(x, y, z);
            this.sceneBundle1.controls.update();
            for(let i = 0; i < this.wasmInstance.get_number_of_meshes(); i++) {
                const wO = this.wasmInstance.get_mesh_on_idx(i);
                const mesh = this.meshes.get(i);
                if(!wO || !mesh) continue;
                const matrix = new Float32Array(wO.get_matrix());
                const threeMatrix = new THREE.Matrix4();
                threeMatrix.fromArray(matrix);
                threeMatrix.transpose();
                mesh.matrix = threeMatrix;
            }
            this.sceneBundle1.scene.updateMatrixWorld(true);
            this.renderer.render(this.sceneBundle1.scene, this.sceneBundle1.camera);
        }

        let t = 0;
        const stepPositioning = () => {
            t = this.wasmInstance.OnLeftButtonUpCustom([cameraFocalPoint.x, cameraFocalPoint.y, cameraFocalPoint.z], update_focal_point_lambda, t);
            if (t < 1.0) {
                requestAnimationFrame(stepPositioning);
            }
            this.updateMeshes()
            if (t < 1) {
                event.stopPropagation();
            }
        }
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

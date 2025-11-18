import { MainModule } from './webAssembly/build/dist/native.js';
import { fitCameraToAllMeshesFromZPlane } from './utils/sceneUtils';
import {
    createCurveLineObjects,
    createBasicCurveLineObjects,
    curvesToCurvePolyOpts,
    updateBasicCurveLines,
    getMeshFromWasm, createMesh, buffersToCurves, curvesToCurveLines
} from "@/utils/utlis";
import { TestClassThreejsBase } from "@/TestThreejsBase";

export class IsobandsExample extends TestClassThreejsBase {
    private curveObjects: any;
    private useBasicLines = false;
    private keyPressed = false;

    constructor(wasmInstance: MainModule, paths: string[], container: HTMLElement, container2?: HTMLElement) {
        super(wasmInstance, paths, container, container2, true);

        const interceptKeyDownEvent = (event: KeyboardEvent) => {
            if (this.keyPressed) return;
            this.keyPressed = true;
            wasmInstance.OnKeyPress(event.key);
            this.updateMeshes();
        };
        const interceptKeyUpEvent = (_event: KeyboardEvent) => {
            this.keyPressed = false;
        };
        window.addEventListener('keydown', interceptKeyDownEvent);
        window.addEventListener('keyup', interceptKeyUpEvent);


        const interceptWheelEvent = (event: WheelEvent) => {
            if (event.deltaX === 0) {
                // Do nothing if deltaX is zero to avoid division by zero
                return;
            }
            const absDelta = event.deltaX / Math.abs(event.deltaX);
            wasmInstance.OnMouseWheel(absDelta, event.shiftKey);
            this.updateMeshes();
        };
        window.addEventListener('wheel', interceptWheelEvent);

        const opts: curvesToCurvePolyOpts = {
            tubeColor: 0x2020ff,    // Blue lines for isobands curves
            lineWidth: 0.2
        };
        this.curveObjects = this.useBasicLines ? createBasicCurveLineObjects(opts) : createCurveLineObjects(opts);
        this.sceneBundle1.scene.add(this.curveObjects.lines);
        if (this.sceneBundle2 && this.renderer2) {
            const mesh = createMesh();
            this.meshes2.set(0, mesh)
            this.sceneBundle2.scene.add(mesh);
        }
        this.updateMeshes();
        fitCameraToAllMeshesFromZPlane(this.sceneBundle1);
    }

    public updateMeshes() {
        super.updateMeshes();

        // Update curve mesh (intersection curves)
        const cO = this.wasmInstance.get_curve_mesh()
        if (cO && cO.polydata_updated) {
            const points = cO.get_curve_points();
            const ids = cO.get_curve_ids();
            const offsets = cO.get_curve_offsets();
            const lines = buffersToCurves(points, ids, offsets);
            if (this.useBasicLines) {
                updateBasicCurveLines(lines, this.curveObjects);
            } else {
                curvesToCurveLines(lines, this.curveObjects, {samplesPerSegment: 3, tension: 0.5});
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
        this.wasmInstance.run_main_isobands(this.paths[0]);
        this.wasmInstance.FS.unlink(this.paths[0]);
    }
}


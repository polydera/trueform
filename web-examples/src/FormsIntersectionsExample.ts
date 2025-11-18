import { MainModule } from './webAssembly/build/dist/native.js'
import { fitCameraToAllMeshesFromZPlane } from './utils/sceneUtils';
import {
    buffersToCurves, createCurveLineObjects,
    CurveLineObjects, curvesToCurveLines, curvesToCurvePolyOpts,
    getMeshFromWasm, createBasicCurveLineObjects, updateBasicCurveLines
} from "@/utils/utlis";
import {TestClassThreejsBase} from "@/TestThreejsBase";

export class FormsIntersectionsExample extends TestClassThreejsBase {
    private curveObjects: CurveLineObjects | any;
    private useBasicLines = false; // Set to true to use basic LineSegments for debugging
    private keyPressed = false;

    constructor(wasmInstance: MainModule, paths: string[], container: HTMLElement) {
        super(wasmInstance, paths, container, undefined, true);

        // Add keyboard event listeners
        const interceptKeyDownEvent = (event: KeyboardEvent) => {
            if(this.keyPressed) return;
            this.keyPressed = true;
            this.wasmInstance.OnKeyPress(event.key)
            this.updateMeshes()
        }
        const interceptKeyUpEvent = (_event: KeyboardEvent) => {
            this.keyPressed = false;
        }
        window.addEventListener('keydown', interceptKeyDownEvent);
        window.addEventListener('keyup', interceptKeyUpEvent);


        // Setup curve rendering
        const opts: curvesToCurvePolyOpts = {
            tubeColor: 0xff2020,    // Red lines for intersection curves
            lineWidth: 0.2
        };

        if (this.useBasicLines) {
            this.curveObjects = createBasicCurveLineObjects(opts);
        } else {
            this.curveObjects = createCurveLineObjects(opts);
        }
        this.sceneBundle1.scene.add(this.curveObjects.lines);

        this.updateMeshes();
        fitCameraToAllMeshesFromZPlane(this.sceneBundle1)
    }

    public runMain(){
        const v = new this.wasmInstance.VectorString()
        for(let i = 0; i < this.paths.length; i++) {
            v.push_back(this.paths[i]);
        }
        this.wasmInstance.run_main_forms_intersections(v);
        for(let i = 0; i < this.paths.length; i++) {
            this.wasmInstance.FS.unlink(this.paths[i]);
        }
    }

    public updateMeshes() {
        super.updateMeshes();

        // Update curve mesh (intersection curves)
        const cO = this.wasmInstance.get_curve_mesh()
        if(cO && cO.polydata_updated) {
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
    }
}

import { MainModule } from './webAssembly/dist/native.js'
import * as THREE from "three";
import Stats from 'stats-gl';
import {createSceneWithCustomConfig, SceneBundle, createBidirectionalSyncedScenes} from './utils/sceneUtils';
import {
    buffersToCurves, createCurveLineObjects,
    createMesh, CurveLineObjects, curvesToCurveLines, curvesToCurveLinesFast, curvesToCurvePolyOpts,
    getMeshFromWasm, createBasicCurveLineObjects, updateBasicCurveLines
} from "@/utils/utlis";


export class TestClassThreejs {
    private readonly wasmInstance: MainModule;

    // First renderer (primary scene)
    private readonly renderer: THREE.WebGLRenderer;
    private readonly sceneBundle1: SceneBundle;
    private meshes = new Map<number, THREE.Mesh>()
    private curveObjects: CurveLineObjects | any;
    private useBasicLines = false; // Set to true to use basic LineSegments for debugging

    // Second renderer (secondary scene)
    private readonly renderer2?: THREE.WebGLRenderer;
    private readonly sceneBundle2?: SceneBundle;
    private meshes2 = new Map<number, THREE.Mesh>()
    private keyPressed = false;
    private stats = new Stats({horizontal: false, trackGPU: true});

    private renderer2Interactive = false;

    constructor(wasmInstance: MainModule, path: string, container: HTMLElement, container2?: HTMLElement) {
        this.wasmInstance = wasmInstance;

        // Setup first renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true } );
        this.renderer.setPixelRatio( window.devicePixelRatio );
        this.renderer.setClearColor( 0x000000, 0.0 );
        const rect = container.getBoundingClientRect();
        this.renderer.setSize(rect.width, rect.height);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.innerHTML = "";
        container.appendChild(this.renderer.domElement);
        this.stats.init(this.renderer);
        container.appendChild( this.stats.dom );

        // Setup second renderer if container2 is provided
        if (container2) {
            this.renderer2 = new THREE.WebGLRenderer({ antialias: true } );
            this.renderer2.setPixelRatio( window.devicePixelRatio );
            const rect2 = container2.getBoundingClientRect();
            this.renderer2.setSize(rect2.width, rect2.height);
            this.renderer2.shadowMap.enabled = true;
            this.renderer2.shadowMap.type = THREE.PCFSoftShadowMap;
            container2.innerHTML = "";
            container2.appendChild(this.renderer2.domElement);
        }

        //////////////////////////// Scene Setup Using Utility Functions //////////////////////////////////////
        // Create synchronized scenes if we have both renderers
        if (this.renderer2) {
            const config1 = {
                backgroundColor: 0x222222,
                cameraPosition: { x: 0, y: 50, z: 0 },
                cameraLookAt: { x: 0, y: 0, z: 0 },
                ambientLightIntensity: 0.8,
                directionalLightIntensity: 0.8,
                enableShadows: true
            };
            const config2 = {
                backgroundColor: 0x333333,
                cameraPosition: { x: 0, y: 50, z: 25 },
                cameraLookAt: { x: 0, y: 0, z: 0 },
                ambientLightIntensity: 0.8,
                directionalLightIntensity: 0.8,
                enableShadows: true
            };

            // Use bidirectional synchronized scenes (interaction on either renderer affects both)
            const { sceneBundle1, sceneBundle2 } = createBidirectionalSyncedScenes(this.renderer, this.renderer2, config1, config2);
            this.sceneBundle1 = sceneBundle1;
            this.sceneBundle2 = sceneBundle2;
        } else {
            // Create first scene with camera, controls, and lighting (single renderer mode)
            this.sceneBundle1 = createSceneWithCustomConfig(this.renderer, 1);
        }

        this.animate();

        // Add resize event listener
        window.addEventListener('resize', () => {
            const rect = container.getBoundingClientRect();
            this.renderer.setSize(rect.width, rect.height);
            this.sceneBundle1.camera.aspect = rect.width / rect.height;
            this.sceneBundle1.camera.updateProjectionMatrix();
            this.sceneBundle1.controls.update();
            this.renderer.render(this.sceneBundle1.scene, this.sceneBundle1.camera);

            // Handle second renderer resize
            if (this.renderer2 && this.sceneBundle2 && container2) {
                const rect2 = container2.getBoundingClientRect();
                this.renderer2.setSize(rect2.width, rect2.height);
                this.sceneBundle2.camera.aspect = rect2.width / rect2.height;
                this.sceneBundle2.camera.updateProjectionMatrix();
                this.sceneBundle2.controls.update();
                this.renderer2.render(this.sceneBundle2.scene, this.sceneBundle2.camera);
            }
        });

        const raycaster = new THREE.Raycaster();
        const ndc = new THREE.Vector2();
        const ray = new THREE.Ray();
        // Add event interception for pointer events
        const interceptEvent = (event: PointerEvent) => {
            // Get bounding rect and mouse position

            const rect = this.renderer.domElement.getBoundingClientRect();
            ndc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            ndc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            // Build world ray
            raycaster.setFromCamera(ndc, this.sceneBundle1.camera);
            ray.copy(raycaster.ray);
            // 2) reusable math objects
            const cameraPosition = this.sceneBundle1.camera.position.clone();
            const dir = new THREE.Vector3();
            this.sceneBundle1.camera.getWorldDirection(dir);
            const cameraFocalPoint = cameraPosition.clone().add(dir.multiplyScalar(100));
            let handled = false;
            if (event.type === 'pointermove') {
                // console.log("pointermove ray", cameraPosition, cameraFocalPoint, ray.origin, ray.direction)
                const v1 = ray.origin.clone()
                const v2 = ray.direction.clone()
                const v3 = cameraPosition
                const v4 = cameraFocalPoint
                handled = this.wasmInstance.OnMouseMove(
                    [v1.x, v1.y, v1.z],
                    [v2.x, v2.y, v2.z],
                    [v3.x, v3.y, v3.z],
                    [v4.x, v4.y, v4.z]);
            } else if (event.type === 'pointerdown') {
                handled = this.wasmInstance.OnLeftButtonDown();
            } else if (event.type === 'pointerup') {
                handled = this.wasmInstance.OnLeftButtonUp();
            }
            this.updateMeshes()
            if (handled) {
                event.stopPropagation();
            }
        };
        this.renderer.domElement.addEventListener('pointerdown', interceptEvent, true);
        this.renderer.domElement.addEventListener('pointermove', interceptEvent, true);
        this.renderer.domElement.addEventListener('pointerup', interceptEvent, true);

        // Add event listeners to second renderer if it exists
        if (this.renderer2 && this.renderer2Interactive) {
            this.renderer2.domElement.addEventListener('pointerdown', interceptEvent, true);
            this.renderer2.domElement.addEventListener('pointermove', interceptEvent, true);
            this.renderer2.domElement.addEventListener('pointerup', interceptEvent, true);
        }
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

        this.wasmInstance.run_main(path);
        this.wasmInstance.FS.unlink(path);

        for(let i = 0; i < 2; i++) {
            const mesh = createMesh();
            this.meshes.set(i, mesh)
            this.sceneBundle1.scene.add(mesh);
        }

        const opts: curvesToCurvePolyOpts = {
            tubeColor: 0x00ff88,    // Green lines
            lineWidth: 0.2
        };

        if (this.useBasicLines) {
            // Use basic LineSegments for compatibility
            this.curveObjects = createBasicCurveLineObjects(opts);
            console.log('Using basic LineSegments for curve rendering');
        } else {
            this.curveObjects = createCurveLineObjects(opts);
            console.log('Using LineSegments2 for curve rendering');
        }
        this.sceneBundle1.scene.add(this.curveObjects.lines);


        if(this.sceneBundle2 && this.renderer2) {
            const mesh = createMesh();
            this.meshes2.set(0, mesh)
            this.sceneBundle2.scene.add(mesh);
        }
        this.updateMeshes();
        // const m = this.meshes.get(0)
        // if(m) fitCameraToObject(this.sceneBundle1.camera, m, 1);
        // const m2 = this.meshes2.get(0)
        // if(m2 && this.sceneBundle2) fitCameraToObject(this.sceneBundle2.camera, m2, 1);
    }

    private updateMeshes(){
        for(let i = 0; i < 2; i++) {
            const wO = this.wasmInstance.get_mesh_on_idx(i);
            const mesh = this.meshes.get(i);
            if(!wO || !mesh) continue;
            getMeshFromWasm(wO, mesh);
        }

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
                // curvesToCurveLinesFast(lines, this.curveObjects, {closed: true});
            }
        }

        if (this.renderer2 && this.sceneBundle2) {
            const wO = this.wasmInstance.get_result_mesh();
            const mesh = this.meshes2.get(0);
            if(wO && mesh){
                getMeshFromWasm(wO, mesh);
            }
            // const m2 = this.meshes2.get(0)
            // if(m2 && this.sceneBundle2) fitCameraToObject(this.sceneBundle2.camera, m2, 1);
        }
    }

    // TODO SetMode --> mode will switch the function in OnMouseMove
    // mode will also handle the layout and mesh views in JS

    public getAverageTime(){
        return this.wasmInstance.get_average_time();
    }

    private animate = () => {
        requestAnimationFrame(this.animate);
        this.sceneBundle1.controls.update();
        this.renderer.render(this.sceneBundle1.scene, this.sceneBundle1.camera);
        if (this.renderer2 && this.sceneBundle2) {
            this.sceneBundle2.controls.update();
            this.renderer2.render(this.sceneBundle2.scene, this.sceneBundle2.camera);
        }
        this.stats.update();
    };
}

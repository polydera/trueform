import { MainModule } from './webAssembly/dist/native.js'
import * as THREE from "three";
import Stats from 'stats-gl';
import {createSceneWithCustomConfig, fitCameraToAllMeshesFromZPlane, SceneBundle} from './utils/sceneUtils';
import { createMesh, getMeshFromWasm } from "@/utils/utlis";


export abstract class TestClassThreejsBase {
    protected readonly wasmInstance: MainModule;
    protected paths: string[];

    // First renderer (primary scene)
    protected readonly renderer: THREE.WebGLRenderer;
    protected readonly sceneBundle1: SceneBundle;
    protected meshes = new Map<number, THREE.Mesh>()
    protected stats = new Stats({horizontal: false, trackGPU: true});

    constructor(wasmInstance: MainModule, paths: string[], container: HTMLElement) {
        this.wasmInstance = wasmInstance;
        this.paths = paths;

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
        container.style.position = 'relative';
        this.stats.dom.style.position = 'absolute';
        container.appendChild( this.stats.dom );

        // Create first scene with camera, controls, and lighting (single renderer mode)
        this.sceneBundle1 = createSceneWithCustomConfig(this.renderer, 1);

        this.animate();

        // Add resize event listener
        window.addEventListener('resize', () => {
            const rect = container.getBoundingClientRect();
            this.renderer.setSize(rect.width, rect.height);
            this.sceneBundle1.camera.aspect = rect.width / rect.height;
            this.sceneBundle1.camera.updateProjectionMatrix();
            this.sceneBundle1.controls.update();
            this.renderer.render(this.sceneBundle1.scene, this.sceneBundle1.camera);

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
            if (event.type === 'pointermove' && (event.buttons === 0 || event.buttons === 1)) {
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
            } else if (event.type === 'pointerdown' && event.buttons === 1) {
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

        this.runMain();

        for(let i = 0; i < this.wasmInstance.get_number_of_meshes(); i++) {
            const mesh = createMesh();
            this.meshes.set(i, mesh)
            this.sceneBundle1.scene.add(mesh);
        }

        this.updateMeshes();
        fitCameraToAllMeshesFromZPlane(this.sceneBundle1)
    }

    private updateMeshes(){
        for(let i = 0; i < this.wasmInstance.get_number_of_meshes(); i++) {
            const wO = this.wasmInstance.get_mesh_on_idx(i);
            const mesh = this.meshes.get(i);
            if(!wO || !mesh) continue;
            getMeshFromWasm(wO, mesh);
        }
    }

    public getAverageTime(){
        return this.wasmInstance.get_average_time();
    }

    public getAveragePickTime(){
        return this.wasmInstance.get_average_pick_time();
    }

    abstract runMain(): void;

    private animate = () => {
        requestAnimationFrame(this.animate);
        this.sceneBundle1.controls.update();
        this.renderer.render(this.sceneBundle1.scene, this.sceneBundle1.camera);
        this.stats.update();
    };
}

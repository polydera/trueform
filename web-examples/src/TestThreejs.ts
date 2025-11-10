import { MainModule } from './webAssembly/dist/native.js'
import * as THREE from "three";
import { createSceneWithCustomConfig, SceneBundle } from './utils/sceneUtils';


export class TestClassThreejs {
    private readonly wasmInstance: MainModule;

    // First renderer (primary scene)
    private readonly renderer: THREE.WebGLRenderer;
    private readonly sceneBundle1: SceneBundle;
    public material = new THREE.Material();
    private meshes = new Map<number, THREE.Mesh>()

    // Second renderer (secondary scene)
    private readonly renderer2?: THREE.WebGLRenderer;
    private readonly sceneBundle2?: SceneBundle;
    private meshes2 = new Map<number, THREE.Mesh>()
    private keyPressed = false;

    constructor(wasmInstance: MainModule, path: string, container: HTMLElement, container2?: HTMLElement) {
        this.wasmInstance = wasmInstance;

        // Setup first renderer
        this.renderer = new THREE.WebGLRenderer();
        const rect = container.getBoundingClientRect();
        this.renderer.setSize(rect.width, rect.height);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.innerHTML = "";
        container.appendChild(this.renderer.domElement);

        // Setup second renderer if container2 is provided
        if (container2) {
            this.renderer2 = new THREE.WebGLRenderer();
            const rect2 = container2.getBoundingClientRect();
            this.renderer2.setSize(rect2.width, rect2.height);
            this.renderer2.shadowMap.enabled = true;
            this.renderer2.shadowMap.type = THREE.PCFSoftShadowMap;
            container2.innerHTML = "";
            container2.appendChild(this.renderer2.domElement);
        }

        //////////////////////////// Scene Setup Using Utility Functions //////////////////////////////////////
        // Create first scene with camera, controls, and lighting
        this.sceneBundle1 = createSceneWithCustomConfig(this.renderer, 1);
        this.material = new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide, flatShading: true });

        // Create second scene if second renderer exists
        if (this.renderer2) {
            this.sceneBundle2 = createSceneWithCustomConfig(this.renderer2, 2);
        }

        requestAnimationFrame(this.animate);

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
        const interceptKeyDownEvent = (event: KeyboardEvent) => {
            if(this.keyPressed) return;
            this.keyPressed = true;
            this.wasmInstance.OnKeyPress(event.key)
        }
        const interceptKeyUpEvent = (event: KeyboardEvent) => {
            this.keyPressed = false;
        }
        window.addEventListener('keydown', interceptKeyDownEvent);
        window.addEventListener('keyup', interceptKeyUpEvent);


        console.log("TestThree JS 0");
        this.wasmInstance.run_main(path);
        this.wasmInstance.FS.unlink(path);
        console.log("TestThree JS 1");

        for(let i = 0; i < 2; i++) {
            const geometry = new THREE.BufferGeometry();
            const mesh = new THREE.Mesh(geometry, this.material);
            mesh.matrixAutoUpdate = false;
            this.meshes.set(i, mesh)
            this.sceneBundle1.scene.add(mesh);
        }

        if(this.sceneBundle2 && this.renderer2) {
            const geometry = new THREE.BufferGeometry();
            const mesh = new THREE.Mesh(geometry, this.material);
            mesh.matrixAutoUpdate = false;
            this.meshes2.set(0, mesh)
            this.sceneBundle2.scene.add(mesh);
        }
        this.updateMeshes();
    }

    private updateMeshes(){
        let r1NeedsUpdate = false;
        let r2NeedsUpdate = false;
        for(let i = 0; i < 2; i++) {
            const wO = this.wasmInstance.GetMeshOnIdx(i);
            const pU = wO.polydataUpdated;
            const mU = wO.matrixUpdated;
            if(pU || mU) {
                r1NeedsUpdate = true;
                const mesh = this.meshes.get(i);
                if (mesh) {
                    if(pU) {
                        const geometry = mesh.geometry;
                        geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.GetPoints()), 3));
                        geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.GetPolys()), 1));
                    }
                    if(mU) {
                        const matrix = new Float32Array(wO.matrix);
                        matrix[12] = 0;
                        matrix[13] = 0;
                        matrix[14] = 0;
                        matrix[15] = 1;
                        const threeMatrix = new THREE.Matrix4();
                        threeMatrix.fromArray(matrix);
                        threeMatrix.transpose();
                        mesh.matrix = threeMatrix;
                    }
                }
            }
        }
        // TODO do I need this --> update is in animate loop
        if(r1NeedsUpdate)
            this.renderer.render(this.sceneBundle1.scene, this.sceneBundle1.camera);

        if (this.renderer2 && this.sceneBundle2) {
            const wO = this.wasmInstance.GetResultMesh();
            const pU = wO.polydataUpdated;
            const mU = wO.matrixUpdated;
            if(pU || mU) {
                r2NeedsUpdate = true;
                const mesh = this.meshes2.get(0);
                if (mesh) {
                    if(pU) {
                        const geometry = mesh.geometry;
                        geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.GetPoints()), 3));
                        geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.GetPolys()), 1));
                    }
                    if(mU) {
                        const matrix = new Float32Array(wO.matrix);
                        const threeMatrix = new THREE.Matrix4();
                        threeMatrix.fromArray(matrix);
                        threeMatrix.transpose();
                        mesh.matrix = threeMatrix;
                    }
                }
            }
            // TODO do I need this --> update is in animate loop
            if(r2NeedsUpdate)
                this.renderer2.render(this.sceneBundle2.scene, this.sceneBundle2.camera);
        }
    }

    private animate = () => {
        this.sceneBundle1.controls.update();
        this.renderer.render(this.sceneBundle1.scene, this.sceneBundle1.camera);
        if (this.renderer2 && this.sceneBundle2) {
            this.sceneBundle2.controls.update();
            this.renderer2.render(this.sceneBundle2.scene, this.sceneBundle2.camera);
        }
    };
}

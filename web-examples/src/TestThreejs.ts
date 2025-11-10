import { MainModule } from './webAssembly/dist/native.js'
import * as THREE from "three";
import {OrbitControls} from "three/examples/jsm/controls/OrbitControls";


export class TestClassThreejs {
    private readonly wasmInstance: MainModule;// First renderer (primary scene)
    private readonly scene: THREE.Scene;
    private readonly renderer: THREE.WebGLRenderer;
    private readonly light: THREE.DirectionalLight;
    private readonly controls: OrbitControls;
    private readonly camera: THREE.Camera;
    public material = new THREE.Material();
    private meshes = new Map<number, THREE.Mesh>()

    // Second renderer (secondary scene)
    private readonly scene2?: THREE.Scene;
    private readonly renderer2?: THREE.WebGLRenderer;
    private readonly controls2?: OrbitControls;
    private readonly camera2?: THREE.Camera;
    private meshes2 = new Map<number, THREE.Mesh>()

    constructor(wasmInstance: MainModule, path: string, container: HTMLElement, container2?: HTMLElement) {
        this.wasmInstance = wasmInstance;
        console.log("wasmInstance", this.wasmInstance);

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

        //////////////////////////// First Scene Setup //////////////////////////////////////
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x222222); // Dark gray background
        this.material = new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide, flatShading: true });

        // Set up the first camera
        this.camera = new THREE.PerspectiveCamera(75, this.renderer.domElement.width / this.renderer.domElement.height, 0.1, 1000);
        this.camera.position.set(75, 75, 200);
        this.camera.lookAt(new THREE.Vector3(75, 75, 0));
        this.scene.add(this.camera);

        //////////////////////////// Second Scene Setup //////////////////////////////////////
        if (this.renderer2) {
            this.scene2 = new THREE.Scene();
            this.scene2.background = new THREE.Color(0x333333); // Slightly different background
            // Set up the second camera
            this.camera2 = new THREE.PerspectiveCamera(75, this.renderer2.domElement.width / this.renderer2.domElement.height, 0.1, 1000);
            this.camera2.position.set(-75, 75, 200);
            this.camera2.lookAt(new THREE.Vector3(0, 0, 0));
            this.scene2.add(this.camera2);
        }

        // controls for first renderer
        this.controls = new OrbitControls( this.camera, this.renderer.domElement );
        this.controls.enabled = true
        this.controls.enablePan = true;
        this.controls.enableZoom = true;
        this.controls.enableRotate = true;

        // controls for second renderer
        if (this.renderer2 && this.camera2) {
            this.controls2 = new OrbitControls( this.camera2, this.renderer2.domElement );
            this.controls2.enabled = true;
            this.controls2.enablePan = true;
            this.controls2.enableZoom = true;
            this.controls2.enableRotate = true;
        }

        // PROPER LIGHTING SETUP
        // 1. Ambient light for overall scene illumination (soft, even lighting)
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4); // dim white light
        this.scene.add(ambientLight);

        // 2. Main directional light (acts as sun/key light)
        this.light = new THREE.DirectionalLight(0xffffff, 0.8);
        this.light.position.set(50, 100, 50); // positioned above and to the side
        this.light.target.position.set(0, 0, 0);
        this.light.castShadow = true;

        // Configure shadow properties for better quality
        if (this.light.shadow) {
            this.light.shadow.mapSize.width = 2048;
            this.light.shadow.mapSize.height = 2048;
            this.light.shadow.camera.near = 0.5;
            this.light.shadow.camera.far = 500;

            // Configure orthographic shadow camera bounds
            this.light.shadow.camera.left = -100;
            this.light.shadow.camera.right = 100;
            this.light.shadow.camera.top = 100;
            this.light.shadow.camera.bottom = -100;
        }

        this.scene.add(this.light);
        this.scene.add(this.light.target);

        // 3. Fill light (softer, from opposite side to reduce harsh shadows)
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-30, 50, -30);
        fillLight.target.position.set(0, 0, 0);
        fillLight.castShadow = false;
        this.scene.add(fillLight);
        this.scene.add(fillLight.target);

        // 4. Rim light (subtle backlight for better object definition)
        const rimLight = new THREE.DirectionalLight(0x888888, 0.5);
        rimLight.position.set(-50, 20, -100);
        rimLight.target.position.set(0, 0, 0);
        rimLight.castShadow = false;
        this.scene.add(rimLight);
        this.scene.add(rimLight.target);

        // Setup lighting for second scene
        if (this.scene2) {
            // Create separate light instances for second scene
            const ambientLight2 = new THREE.AmbientLight(0x404040, 0.4);
            this.scene2.add(ambientLight2);

            // Main directional light for second scene
            const light2 = new THREE.DirectionalLight(0xffffff, 0.8);
            light2.position.set(50, 100, 50);
            light2.target.position.set(0, 0, 0);
            light2.castShadow = true;

            // Configure shadow properties
            if (light2.shadow) {
                light2.shadow.mapSize.width = 2048;
                light2.shadow.mapSize.height = 2048;
                light2.shadow.camera.near = 0.5;
                light2.shadow.camera.far = 500;
                light2.shadow.camera.left = -100;
                light2.shadow.camera.right = 100;
                light2.shadow.camera.top = 100;
                light2.shadow.camera.bottom = -100;
            }

            this.scene2.add(light2);
            this.scene2.add(light2.target);

            // Fill light for second scene
            const fillLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
            fillLight2.position.set(-30, 50, -30);
            fillLight2.target.position.set(0, 0, 0);
            fillLight2.castShadow = false;
            this.scene2.add(fillLight2);
            this.scene2.add(fillLight2.target);

            // Rim light for second scene
            const rimLight2 = new THREE.DirectionalLight(0x888888, 0.5);
            rimLight2.position.set(-50, 20, -100);
            rimLight2.target.position.set(0, 0, 0);
            rimLight2.castShadow = false;
            this.scene2.add(rimLight2);
            this.scene2.add(rimLight2.target);
        }

        // Create a Three.js mesh for first scene
        this.camera.castShadow = false;
        this.camera.receiveShadow = true;
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
        if (this.renderer2 && this.scene2 && this.camera2) {
            this.controls2?.update();
            this.renderer2.render(this.scene2, this.camera2);
        }
        this.animate(); // Start the animation loop

        // Add resize event listener
        window.addEventListener('resize', () => {
            const rect = container.getBoundingClientRect();
            this.renderer.setSize(rect.width, rect.height);
            if (this.camera instanceof THREE.PerspectiveCamera) {
                this.camera.aspect = rect.width / rect.height;
                this.camera.updateProjectionMatrix();
            }
            this.controls.update();
            this.renderer.render(this.scene, this.camera);

            // Handle second renderer resize
            if (this.renderer2 && this.camera2 && container2) {
                const rect2 = container2.getBoundingClientRect();
                this.renderer2.setSize(rect2.width, rect2.height);
                if (this.camera2 instanceof THREE.PerspectiveCamera) {
                    this.camera2.aspect = rect2.width / rect2.height;
                    this.camera2.updateProjectionMatrix();
                }
                this.controls2?.update();
                this.renderer2.render(this.scene2!, this.camera2);
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
            raycaster.setFromCamera(ndc, this.camera as THREE.PerspectiveCamera);
            ray.copy(raycaster.ray);
            // 2) reusable math objects
            const cameraPosition = (this.camera as THREE.PerspectiveCamera).position.clone();
            let cameraFocalPoint: THREE.Vector3;
            if (this.camera instanceof THREE.PerspectiveCamera) {
                const dir = new THREE.Vector3();
                this.camera.getWorldDirection(dir);
                cameraFocalPoint = cameraPosition.clone().add(dir.multiplyScalar(100));
            } else {
                cameraFocalPoint = new THREE.Vector3();
            }
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


        console.log("TestThree JS 0");
        this.wasmInstance.run_main(path);
        console.log("TestThree JS 1");

        for(let i = 0; i < 2; i++) {
            const wO = this.wasmInstance.GetMeshOnIdx(i);
            const geometry = new THREE.BufferGeometry();
            const mesh = new THREE.Mesh(geometry, this.material);
            this.meshes.set(i, mesh)
            geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.GetPoints()), 3));
            geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.GetPolys()), 1));
            const matrix = new Float32Array(wO.matrix);
            const threeMatrix = new THREE.Matrix4();
            threeMatrix.fromArray(matrix);
            threeMatrix.transpose();
            mesh.applyMatrix4(threeMatrix);
            this.scene.add(mesh);
        }
        this.renderer.render(this.scene, this.camera);
        if (this.renderer2 && this.scene2 && this.camera2) {
            this.renderer2.render(this.scene2, this.camera2);
        }

        this.wasmInstance.FS.unlink(path);

    }

    private updateMeshes(){
        for(let i = 0; i < 2; i++) {
            const wO = this.wasmInstance.GetMeshOnIdx(i);
            console.log("wO", wO)

            // Update mesh in first scene
            const mesh = this.meshes.get(i);
            if(mesh) {
                const geometry = mesh.geometry;
                geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.GetPoints()), 3));
                geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.GetPolys()), 1));
                const matrix = new Float32Array(wO.matrix);
                matrix[12] = 0;
                matrix[13] = 0;
                matrix[14] = 0;
                matrix[15] = 1;
                const threeMatrix = new THREE.Matrix4();
                threeMatrix.fromArray(matrix);
                threeMatrix.transpose();
                mesh.matrixAutoUpdate = false;
                mesh.matrix = threeMatrix;
            }

            // Update mesh in second scene
            const mesh2 = this.meshes2.get(i);
            if(mesh2) {
                const geometry2 = mesh2.geometry;
                geometry2.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.GetPoints()), 3));
                geometry2.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.GetPolys()), 1));
                const matrix = new Float32Array(wO.matrix);
                matrix[12] = 0;
                matrix[13] = 0;
                matrix[14] = 0;
                matrix[15] = 1;
                const threeMatrix = new THREE.Matrix4();
                threeMatrix.fromArray(matrix);
                threeMatrix.transpose();
                mesh2.matrixAutoUpdate = false;
                mesh2.matrix = threeMatrix;
            }
        }

        this.renderer.render(this.scene, this.camera);
        if (this.renderer2 && this.scene2 && this.camera2) {
            this.renderer2.render(this.scene2, this.camera2);
        }
    }

    private animate = () => {
        requestAnimationFrame(this.animate);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);

        // Render second scene if it exists
        if (this.renderer2 && this.scene2 && this.camera2 && this.controls2) {
            this.controls2.update();
            this.renderer2.render(this.scene2, this.camera2);
        }
    };
}

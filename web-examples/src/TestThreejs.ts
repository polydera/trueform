import { MainModule } from './webAssembly/dist/native.js'
import * as THREE from "three";
import {OrbitControls} from "three/examples/jsm/controls/OrbitControls";
import {STLLoader} from "three/examples/jsm/loaders/STLLoader";
import {OBJLoader} from "three/examples/jsm/loaders/OBJLoader";
import {getMeshFromWasm} from "@/utils/utlis";

export class TestClassThreejs {
    private readonly wasmInstance: MainModule;
    private readonly scene: THREE.Scene;
    private readonly renderer: THREE.WebGLRenderer;
    private readonly light: THREE.Light;
    private readonly controls: OrbitControls;
    private readonly camera: THREE.Camera;
    public geometry = new THREE.BufferGeometry();
    public material = new THREE.Material();
    public mesh = new THREE.Mesh(this.geometry, this.material);

    private meshes = new Map<string, THREE.Mesh>()

    constructor(wasmInstance: MainModule, container: HTMLElement, container2?: HTMLElement) {
        this.wasmInstance = wasmInstance;
        console.log("wasmInstance", this.wasmInstance);

        this.renderer = new THREE.WebGLRenderer();
        const rect = container.getBoundingClientRect();
        this.renderer.setSize(rect.width, rect.height); // Fix: use rect.height for correct aspect
        this.renderer.shadowMap.enabled = true;
        container.innerHTML = "";
        container.appendChild(this.renderer.domElement);

        //////////////////////////// tree.js rendering //////////////////////////////////////
        const scene = new THREE.Scene();
        this.scene = scene;
        const material = new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide, flatShading: true });
        this.material = material;

        // this.geometry = new THREE.ConeGeometry(50, 100, 32);
        // this.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(this.wasmObj.outputPoints), 3));
        // this.geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(this.wasmObj.outputPolys), 3));
        // this.mesh.applyMatrix4()


        // const objLoader = new OBJLoader()
        // objLoader.setMaterials( this.material );
        // objLoader.loadAsync("dragon-50k-poisson 1.obj").then((object) => {
        //     console.log("object", object)
        //     this.scene.add(object)
        // })
        const path = "zan0.stl"
        // fetch(path).then(async (response) => {
        //     const aBuff = await response.arrayBuffer();
        //     const intArr = new Int8Array(aBuff);
        //     this.wasmInstance.FS.writeFile(path, intArr);
        //     //
        //     // const geometry = new THREE.BufferGeometry();
        //     // const mesh = new THREE.Mesh(geometry, material);
        //     // this.meshes.set("0", mesh)
        //     // getMeshFromWasm(this.wasmClass, geometry);
        //     // this.scene.add(this.mesh);
        //     // this.renderer.render(this.scene, this.camera);
        //     //
        //     // this.wasmInstance.FS.unlink(path);
        // });

        // const addMesh = (mesh: THREE.Mesh) => {
        //     this.addMesh(mesh)
        // }
        // const loader = new STLLoader()
        // loader.load(
        //     'zan0.stl',
        //     function (geometry) {
        //         addMesh(new THREE.Mesh(geometry, material))
        //     },
        //     (xhr) => {
        //         console.log((xhr.loaded / xhr.total) * 100 + '% loaded')
        //     },
        //     (error) => {
        //         console.log(error)
        //     }
        // )

        console.log("geometry", this.geometry);
        console.log("material", this.material);
        // Set up the camera
        const d = 100;
        this.camera = new THREE.PerspectiveCamera(75, this.renderer.domElement.width / this.renderer.domElement.height, 0.1, 1000);
        this.camera.position.set(75, 75, 200);
        this.camera.lookAt(new THREE.Vector3(75, 75, 0));
        console.log("camera", this.camera);
        this.scene.add(this.camera);

        // controls
        this.controls = new OrbitControls( this.camera, this.renderer.domElement );
        this.controls.enabled = true
        this.controls.enablePan = true;
        this.controls.enableZoom = true;
        this.controls.enableRotate = true;

        // LIGHT
        this.light = new THREE.DirectionalLight(0xffffff, 1);
        this.light.position.set(-75, 0, 0);
        this.light.lookAt(new THREE.Vector3(0,0, 0));
        this.light.visible = true;
        this.light.castShadow = false;
        this.scene.add(this.light);
        const light2 = new THREE.DirectionalLight(0xffffff, 1);
        light2.position.set(75, 0, 0);
        light2.lookAt(new THREE.Vector3(0,0, 0));
        light2.visible = true;
        light2.castShadow = false;
        this.scene.add(light2);
        const light3 = new THREE.DirectionalLight(0xffffff, 1);
        light3.position.set(0, 0, 75 + d * 1.0);
        light3.lookAt(new THREE.Vector3(0, 0, 0));
        light3.visible = true;
        light3.castShadow = false;
        this.scene.add(light3);
        const light4 = new THREE.DirectionalLight(0xffffff, 1);
        light4.position.set(0, 0, -75 + d * 1.0);
        light4.lookAt(new THREE.Vector3(0, 0, 0));
        light4.visible = true;
        light4.castShadow = false;
        this.scene.add(light3);
        const light5 = new THREE.DirectionalLight(0xffffff, 1);
        light5.position.set(0, 75, 0);
        light5.lookAt(new THREE.Vector3(0, 0, 0));
        light5.visible = true;
        light5.castShadow = false;
        this.scene.add(light5);
        const light6 = new THREE.DirectionalLight(0xffffff, 1);
        light6.position.set(0, -75, 0);
        light6.lookAt(new THREE.Vector3(0, 0, 0));
        light6.visible = true;
        light6.castShadow = false;
        this.scene.add(light6);

        // const ambientLight = new THREE.AmbientLight(0xffffff, 2);
        // this.scene.add(ambientLight);

        // Create a Three.js mesh
        this.camera.castShadow = false;
        this.camera.receiveShadow = true;
        this.mesh = new THREE.Mesh(this.geometry, this.material);
        this.mesh.position.set(0, 0, 0);
        this.mesh.castShadow = true;
        this.mesh.receiveShadow = true;

        // Add the mesh to your Three.js scene
        this.scene.add(this.mesh);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
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
        });

        // Add event interception for pointer events
        const interceptEvent = (event: PointerEvent) => {
            // Get bounding rect and mouse position
            const rect = this.renderer.domElement.getBoundingClientRect();
            const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            // Compute world ray
            const pointer = new THREE.Vector2(x, y);
            const raycaster = new THREE.Raycaster();
            raycaster.setFromCamera(pointer, this.camera as THREE.PerspectiveCamera);
            const worldRay = raycaster.ray;
            // Use worldRay.origin and worldRay.direction as needed
            console.log('World Ray:', worldRay.origin, worldRay.direction);
            // TODO send event
            // const alreadyHandled = ;
            // if(alreadyHandled)
            //     event.stopPropagation();
        };
        this.renderer.domElement.addEventListener('pointerdown', interceptEvent, true);
        this.renderer.domElement.addEventListener('pointermove', interceptEvent, true);
        this.renderer.domElement.addEventListener('pointerup', interceptEvent, true);



        console.log("TestThree JS 0");
        this.wasmInstance.run_main(path);
        console.log("TestThree JS 1");

        for(let i = 0; i < 2; i++) {
            const wO = this.wasmInstance.GetMeshOnIdx(i);
            console.log("wO", wO)
            const geometry = new THREE.BufferGeometry();
            const mesh = new THREE.Mesh(geometry, material);
            this.meshes.set("0", mesh)
            geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.GetPoints()), 3));
            console.log("wO.points()", wO.GetPoints())
            console.log("wO.GetPolys()", wO.GetPolys())
            geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.GetPolys()), 1));
            const matrix = wO.matrix;
            console.log("Matrix", matrix)
            const threeMatrix = new THREE.Matrix4();
            threeMatrix.fromArray(matrix);
            mesh.applyMatrix4(threeMatrix);
            this.scene.add(mesh);
        }
        this.renderer.render(this.scene, this.camera);

        this.wasmInstance.FS.unlink(path);

    }

    private addMesh(mesh: THREE.Mesh) {
        // TODO sync mesh to wasm
        // this.wasmInstance
        const pos = mesh.geometry.getAttribute("position")
        console.log("Position", pos)

        const idx = mesh.geometry.getIndex()
        console.log("Index", mesh.geometry.attributes)
        this.scene.add(mesh);
    }

    private animate = () => {
        requestAnimationFrame(this.animate);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    };
}

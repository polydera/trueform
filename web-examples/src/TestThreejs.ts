import * as THREE from "three";
import {OrbitControls} from "three/examples/jsm/controls/OrbitControls";

export class TestClassThreejs {

    private readonly scene: THREE.Scene;
    private readonly renderer: THREE.WebGLRenderer;
    private readonly light: THREE.Light;
    private readonly controls: OrbitControls;
    // private readonly camera: THREE.Camera;
    private readonly camera: THREE.Camera;
    public geometry = new THREE.BufferGeometry();
    public material = new THREE.Material();
    public mesh = new THREE.Mesh(this.geometry, this.material);

    constructor(container: HTMLElement) {
        this.renderer = new THREE.WebGLRenderer();
        const rect = container.getBoundingClientRect();
        this.renderer.setSize(rect.width, rect.height); // Fix: use rect.height for correct aspect
        this.renderer.shadowMap.enabled = true;
        container.innerHTML = "";
        container.appendChild(this.renderer.domElement);

        //////////////////////////// tree.js rendering //////////////////////////////////////
        this.scene = new THREE.Scene();
        // Create Three.js material
        // this.material = new THREE.MeshBasicMaterial( { color: 0x44ff44, side: THREE.DoubleSide} )
        // this.material = new THREE.MeshPhongMaterial({ color: 0x00ff00, side: THREE.DoubleSide });
        this.material = new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide });

        this.geometry = new THREE.ConeGeometry(50, 100, 32);
        // this.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(this.wasmObj.outputPoints), 3));
        // this.geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(this.wasmObj.outputPolys), 3));
        // this.mesh.applyMatrix4()

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
        this.light = new THREE.DirectionalLight(0xffffff, 2);
        this.light.position.set(75 - d * 1.0, 75 - d * 1.0, 0 + d * 1.0);
        this.light.lookAt(new THREE.Vector3(75, 75, 0));
        this.light.visible = true;
        this.light.castShadow = false;
        this.scene.add(this.light);

        const ambientLight = new THREE.AmbientLight(0xffffff, 2);
        this.scene.add(ambientLight);

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
    }

    private animate = () => {
        requestAnimationFrame(this.animate);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    };
}

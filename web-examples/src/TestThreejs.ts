// import { IEngineGenerated } from "@/bin/test/IEngineTestGenerated";

import * as THREE from "three";

export class TestClassThreejs {
    private readonly wasmObj: IEngineGenerated.prototype.xVtkPolyDataJSViewTest;

    private readonly scene: THREE.Scene;
    private readonly renderer: THREE.Renderer;
    private readonly light: THREE.Light;
    // private readonly camera: THREE.Camera;
    private readonly camera: THREE.Camera;
    public geometry = new THREE.BufferGeometry();
    public material = new THREE.Material();
    public mesh = new THREE.Mesh(this.geometry, this.material);

    constructor(wasmObj: IEngineGenerated.prototype.xVtkPolyDataJSViewTest, renderer: THREE.Renderer, size: number) {
        this.wasmObj = wasmObj;

        this.renderer = renderer;

        //////////////////////////// tree.js rendering //////////////////////////////////////
        console.log("wasm 1");
        this.wasmObj.waivingSurfaceTest(size);
        this.wasmObj.ApplyWavingEffectParallelThreeJS(0);
        console.log("wasm 2");

        this.scene = new THREE.Scene();
        // Create Three.js material
        // this.material = new THREE.MeshBasicMaterial( { color: 0x44ff44, side: THREE.DoubleSide} )
        // this.material = new THREE.MeshPhongMaterial({ color: 0x00ff00, side: THREE.DoubleSide });
        this.material = new THREE.MeshLambertMaterial({ color: 0x00ff00, side: THREE.DoubleSide });

        this.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(this.wasmObj.outputPoints), 3));
        this.geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(this.wasmObj.outputPolys), 3));
        // this.mesh.applyMatrix4()

        console.log("geometry", this.geometry);
        console.log("material", this.material);
        // Set up the camera
        const d = 200;
        // this.camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
        this.camera = new THREE.OrthographicCamera(-300, 300, 300, -300);
        // Adjust the camera position
        this.camera.position.set(75 - d * 1.0, 75 - d * 1.0, 0 + d * 1.0);
        // Set the camera's look-at point (focal point)
        this.camera.lookAt(new THREE.Vector3(75, 75, 0));
        // Adjust the camera's up vector
        this.camera.up.set(0, 0, 1);
        console.log("camera", this.camera);
        this.scene.add(this.camera);


        // // controls
        // const controls = new OrbitControls( camera, renderer.domElement );
        // controls.enablePan = false;

        // LIGHT
        this.light = new THREE.PointLight(0xffffff, 2);
        // this.light = new THREE.DirectionalLight(0xffffff, 2);
        this.light.position.set(75 - d * 1.0, 75 - d * 1.0, 0 + d * 1.0);
        this.light.lookAt(new THREE.Vector3(75, 75, 0));
        this.light.visible = true;
        this.light.castShadow = true;

        const ambientLight = new THREE.AmbientLight(0xffffff, 2);
        this.scene.add(ambientLight);

        // const helper = new THREE.CameraHelper(this.light.shadow.camera);
        // this.scene.add(helper);

        this.scene.add(this.light);

        // Create a Three.js mesh
        this.camera.castShadow = true;
        this.camera.receiveShadow = true;
        this.mesh = new THREE.Mesh(this.geometry, this.material);
        this.mesh.position.set(0, 0, 0);
        this.mesh.castShadow = true;
        this.mesh.receiveShadow = true;

        // Add the mesh to your Three.js scene
        this.scene.add(this.mesh);
        this.renderer.render(this.scene, this.camera);
    }

    public run() {
        if (this.wasmObj) {
            let startTime: number | undefined;
            const endT = 4000; // endT = 2s in ms
            let counter = 0;

            const animate = (timestamp: number) => {
                if (!startTime) startTime = timestamp;
                const dt = (timestamp - startTime) | 0;

                if (dt < endT) {
                    counter++;
                    // Update your scene or perform calculations here
                    this.wasmObj.ApplyWavingEffectParallelThreeJS(dt);
                    this.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(this.wasmObj.outputPoints), 3));
                    this.geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(this.wasmObj.outputPolys), 3));
                    this.geometry.computeVertexNormals();
                    // Render the scene
                    this.renderer.render(this.scene, this.camera);

                    // Request the next frame
                    requestAnimationFrame(animate);
                } else {
                    const fps = (counter / endT) * 1000;
                    console.log("FPS was: ", fps);
                }
            };

            // Start the animation loop
            requestAnimationFrame(animate);
        }
    }
}

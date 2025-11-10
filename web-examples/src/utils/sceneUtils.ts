import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

export interface SceneConfig {
    backgroundColor?: number;
    cameraPosition?: { x: number; y: number; z: number };
    cameraLookAt?: { x: number; y: number; z: number };
    ambientLightColor?: number;
    ambientLightIntensity?: number;
    directionalLightColor?: number;
    directionalLightIntensity?: number;
    directionalLightPosition?: { x: number; y: number; z: number };
    enableShadows?: boolean;
}

export interface SceneBundle {
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    directionalLight: THREE.DirectionalLight;
}

export function createScene(
    renderer: THREE.WebGLRenderer,
    config: SceneConfig = {}
): SceneBundle {
    // Default configuration
    const {
        backgroundColor = 0x222222,
        cameraPosition = { x: 75, y: 75, z: 200 },
        cameraLookAt = { x: 0, y: 0, z: 0 },
        ambientLightColor = 0x404040,
        ambientLightIntensity = 0.4,
        directionalLightColor = 0xffffff,
        directionalLightIntensity = 0.8,
        directionalLightPosition = { x: 50, y: 100, z: 50 },
        enableShadows = true
    } = config;

    // Create scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(backgroundColor);

    // Create camera
    const camera = new THREE.PerspectiveCamera(
        75,
        renderer.domElement.width / renderer.domElement.height,
        0.1,
        1000
    );
    camera.position.set(cameraPosition.x, cameraPosition.y, cameraPosition.z);
    camera.lookAt(new THREE.Vector3(cameraLookAt.x, cameraLookAt.y, cameraLookAt.z));
    camera.castShadow = false;
    camera.receiveShadow = true;
    scene.add(camera);

    // Create orbit controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enabled = true;
    controls.enablePan = true;
    controls.enableZoom = true;
    controls.enableRotate = true;

    // Setup lighting
    // 1. Ambient light for overall scene illumination
    const ambientLight = new THREE.AmbientLight(ambientLightColor, ambientLightIntensity);
    scene.add(ambientLight);

    // 2. Main directional light (acts as sun/key light)
    const directionalLight = new THREE.DirectionalLight(directionalLightColor, directionalLightIntensity);
    directionalLight.position.set(directionalLightPosition.x, directionalLightPosition.y, directionalLightPosition.z);
    directionalLight.target.position.set(0, 0, 0);
    directionalLight.castShadow = enableShadows;

    // 2. Main directional light (acts as sun/key light)
    const directionalLight2 = new THREE.DirectionalLight(directionalLightColor, directionalLightIntensity);
    directionalLight2.position.set(-directionalLightPosition.x, -directionalLightPosition.y, -directionalLightPosition.z);
    directionalLight2.target.position.set(0, 0, 0);
    directionalLight2.castShadow = enableShadows;

    // Configure shadow properties for better quality
    if (enableShadows && directionalLight.shadow) {
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 500;
        directionalLight.shadow.camera.left = -100;
        directionalLight.shadow.camera.right = 100;
        directionalLight.shadow.camera.top = 100;
        directionalLight.shadow.camera.bottom = -100;
    }

    scene.add(directionalLight);
    scene.add(directionalLight.target);
    scene.add(directionalLight2);
    scene.add(directionalLight2.target);

    // 3. Fill light (softer, from opposite side to reduce harsh shadows)
    const fillLight = new THREE.DirectionalLight(directionalLightColor, 0.3);
    fillLight.position.set(-30, 50, -30);
    fillLight.target.position.set(0, 0, 0);
    fillLight.castShadow = false;
    scene.add(fillLight);
    scene.add(fillLight.target);

    // 4. Rim light (subtle backlight for better object definition)
    const rimLight = new THREE.DirectionalLight(0x888888, 0.5);
    rimLight.position.set(-50, 20, -100);
    rimLight.target.position.set(0, 0, 0);
    rimLight.castShadow = false;
    scene.add(rimLight);
    scene.add(rimLight.target);

    return {
        scene,
        camera,
        controls,
        directionalLight
    };
}

export function createSceneWithCustomConfig(
    renderer: THREE.WebGLRenderer,
    sceneNumber: number = 1
): SceneBundle {
    const configs: SceneConfig[] = [
        // Scene 1 configuration
        {
            backgroundColor: 0x222222,
            cameraPosition: { x: 75, y: 75, z: 200 },
            cameraLookAt: { x: 75, y: 75, z: 0 },
            ambientLightIntensity: 0.4,
            directionalLightIntensity: 0.8,
            enableShadows: true
        },
        // Scene 2 configuration
        {
            backgroundColor: 0x333333,
            cameraPosition: { x: 75, y: 75, z: 200 },
            cameraLookAt: { x: 75, y: 75, z: 0 },
            ambientLightIntensity: 0.4,
            directionalLightIntensity: 0.8,
            enableShadows: true
        }
    ];

    const config = configs[sceneNumber - 1] || configs[0];
    return createScene(renderer, config);
}

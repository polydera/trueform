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
        cameraPosition = { x: 20, y: 20, z: 30 },
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
            cameraPosition: { x: 25, y: 25, z: 25 },
            cameraLookAt: { x: 0, y: 0, z: 0 },
            ambientLightIntensity: 0.4,
            directionalLightIntensity: 0.8,
            enableShadows: true
        },
        // Scene 2 configuration
        {
            backgroundColor: 0x333333,
            cameraPosition: { x: 25, y: 25, z: 25 },
            cameraLookAt: { x: 0, y: 0, z: 0 },
            ambientLightIntensity: 0.4,
            directionalLightIntensity: 0.8,
            enableShadows: true
        }
    ];

    const config = configs[sceneNumber - 1] || configs[0];
    return createScene(renderer, config);
}


export function fitCameraToObject(camera: THREE.PerspectiveCamera, object: THREE.Mesh, offset = 1.25, controls?: OrbitControls) {
    // Compute the bounding box of the object (or entire scene)
    const box = new THREE.Box3().setFromObject(object);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    // Compute distance needed for the camera to fit the object
    const maxSize = Math.max(size.x, size.y, size.z);
    const fitHeightDistance = maxSize / (2 * Math.atan((Math.PI * camera.fov) / 360));
    const fitWidthDistance = fitHeightDistance / camera.aspect;
    const distance = offset * Math.max(fitHeightDistance, fitWidthDistance);

    // Compute new camera position
    const direction = new THREE.Vector3()
        .subVectors(camera.position, center)
        .normalize()
        .multiplyScalar(distance);

    camera.position.copy(direction.add(center));

    // Update camera near/far planes
    camera.near = distance / 100;
    camera.far = distance * 100;
    camera.updateProjectionMatrix();

    // Optionally re-center controls (if using OrbitControls)
    if (controls) {
        controls.target.copy(center);
        controls.update();
    }
}

/**
 * Synchronizes two OrbitControls so they have the same camera parameters and interactions
 */
export function synchronizeOrbitControls(primaryControls: OrbitControls, secondaryControls: OrbitControls) {
    // Function to copy camera parameters from primary to secondary
    const syncCameras = () => {
        const primaryCamera = primaryControls.object as THREE.PerspectiveCamera;
        const secondaryCamera = secondaryControls.object as THREE.PerspectiveCamera;
        
        // Copy position
        secondaryCamera.position.copy(primaryCamera.position);
        
        // Copy rotation/orientation
        secondaryCamera.quaternion.copy(primaryCamera.quaternion);
        
        // Copy target (look at point)
        secondaryControls.target.copy(primaryControls.target);
        
        // Update matrices
        secondaryCamera.updateMatrixWorld();
        secondaryControls.update();
    };

    // Listen to primary controls changes and sync to secondary
    primaryControls.addEventListener('change', syncCameras);
    
    // Also sync on start and end events for smooth interaction
    primaryControls.addEventListener('start', () => {
        syncCameras();
    });
    
    primaryControls.addEventListener('end', () => {
        syncCameras();
    });

    // Initial sync
    syncCameras();
}

/**
 * Creates a synchronized scene pair where both renderers have synchronized orbit controls
 */
export function createSynchronizedScenes(
    renderer1: THREE.WebGLRenderer,
    renderer2: THREE.WebGLRenderer,
    config1: SceneConfig = {},
    config2: SceneConfig = {}
): { sceneBundle1: SceneBundle; sceneBundle2: SceneBundle } {
    // Create both scenes
    const sceneBundle1 = createScene(renderer1, config1);
    const sceneBundle2 = createScene(renderer2, config2);
    
    // Synchronize the orbit controls (renderer1 is primary)
    synchronizeOrbitControls(sceneBundle1.controls, sceneBundle2.controls);
    
    return { sceneBundle1, sceneBundle2 };
}

/**
 * Alternative approach: Create scene with shared camera parameters
 * This creates a bidirectional sync where interaction on either renderer affects both
 */
export function createBidirectionalSyncedScenes(
    renderer1: THREE.WebGLRenderer,
    renderer2: THREE.WebGLRenderer,
    config1: SceneConfig = {},
    config2: SceneConfig = {}
): { sceneBundle1: SceneBundle; sceneBundle2: SceneBundle } {
    // Create both scenes
    const sceneBundle1 = createScene(renderer1, config1);
    const sceneBundle2 = createScene(renderer2, config2);
    
    let isSyncing = false; // Prevent infinite loops
    
    // Function to sync from source to target
    const syncControls = (sourceControls: OrbitControls, targetControls: OrbitControls) => {
        if (isSyncing) return;
        isSyncing = true;
        
        const sourceCamera = sourceControls.object as THREE.PerspectiveCamera;
        const targetCamera = targetControls.object as THREE.PerspectiveCamera;
        
        // Copy camera parameters
        targetCamera.position.copy(sourceCamera.position);
        targetCamera.quaternion.copy(sourceCamera.quaternion);
        targetControls.target.copy(sourceControls.target);
        
        // Update matrices
        targetCamera.updateMatrixWorld();
        targetControls.update();
        
        isSyncing = false;
    };
    
    // Setup bidirectional sync
    const setupControlsSync = (controls1: OrbitControls, controls2: OrbitControls) => {
        const syncEvents = ['change', 'start', 'end'];
        
        syncEvents.forEach(eventType => {
            controls1.addEventListener(eventType, () => {
                syncControls(controls1, controls2);
            });
            
            controls2.addEventListener(eventType, () => {
                syncControls(controls2, controls1);
            });
        });
    };
    
    setupControlsSync(sceneBundle1.controls, sceneBundle2.controls);
    
    // Initial sync (use sceneBundle1 as initial source)
    syncControls(sceneBundle1.controls, sceneBundle2.controls);
    
    return { sceneBundle1, sceneBundle2 };
}

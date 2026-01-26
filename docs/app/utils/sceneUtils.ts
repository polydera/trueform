import * as THREE from "three";
import { ArcballControls } from "three/addons/controls/ArcballControls.js";

export interface SceneConfig {
    backgroundColor?: number;
    enableFog?: boolean;
    fogNear?: number;
    fogFar?: number;
    cameraPosition?: { x: number; y: number; z: number };
    cameraLookAt?: { x: number; y: number; z: number };
    ambientLightColor?: number;
    ambientLightIntensity?: number;
    hemisphereLightSkyColor?: number;
    hemisphereLightGroundColor?: number;
    hemisphereLightIntensity?: number;
    directionalLightColor?: number;
    directionalLightIntensity?: number;
    directionalLightPosition?: { x: number; y: number; z: number };
    fillLightColor?: number;
    fillLightIntensity?: number;
    rimLightColor?: number;
    rimLightIntensity?: number;
    enableShadows?: boolean;
}

export interface SceneBundle {
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: ArcballControls;
    directionalLight: THREE.DirectionalLight;
}

export function createScene(
    renderer: THREE.WebGLRenderer,
    config: SceneConfig = {}
): SceneBundle {
    // Default configuration
    const {
        backgroundColor = 0x202020,
        enableFog = true,
        fogNear = 80,
        fogFar = 450,
        cameraPosition = { x: 20, y: 20, z: 30 },
        cameraLookAt = { x: 0, y: 0, z: 0 },
        ambientLightColor = 0x404040,
        ambientLightIntensity = 0.4,
        hemisphereLightSkyColor = 0xf2f5ff,
        hemisphereLightGroundColor = 0x1a1614,
        hemisphereLightIntensity = 0.6,
        directionalLightColor = 0xffffff,
        directionalLightIntensity = 0.9,
        directionalLightPosition = { x: 60, y: 120, z: 80 },
        fillLightColor = 0xfefaf3,
        fillLightIntensity = 0.25,
        rimLightColor = 0xdde6ff,
        rimLightIntensity = 0.25,
        enableShadows = true
    } = config;

    // Create scene
    const scene = new THREE.Scene();
    const background = new THREE.Color(backgroundColor);
    scene.background = background;
    if (enableFog) {
        scene.fog = new THREE.Fog(background.clone(), fogNear, fogFar);
    } else {
        scene.fog = null;
    }

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

    // Create trackball controls
    const controls = new ArcballControls(camera, renderer.domElement, scene);
    controls.rotateSpeed = 1.2;
    controls.enableGizmos = true;
    controls.setGizmosVisible(false);

    // Setup lighting
    // 1. Ambient light for overall scene illumination
    const ambientLight = new THREE.AmbientLight(ambientLightColor, ambientLightIntensity);
    scene.add(ambientLight);

    const hemisphereLight = new THREE.HemisphereLight(
        hemisphereLightSkyColor,
        hemisphereLightGroundColor,
        hemisphereLightIntensity
    );
    hemisphereLight.position.set(0, 200, 0);
    scene.add(hemisphereLight);

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
        const shadowCam = directionalLight.shadow.camera as THREE.OrthographicCamera;
        directionalLight.shadow.mapSize.width = 4096;
        directionalLight.shadow.mapSize.height = 4096;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 900;
        const range = 220;
        shadowCam.left = -range;
        shadowCam.right = range;
        shadowCam.top = range;
        shadowCam.bottom = -range;
        directionalLight.shadow.bias = -0.0002;
        directionalLight.shadow.normalBias = 0.01;
    }

    scene.add(directionalLight);
    scene.add(directionalLight.target);
    scene.add(directionalLight2);
    scene.add(directionalLight2.target);

    // 3. Fill light (softer, from opposite side to reduce harsh shadows)
    const fillLight = new THREE.DirectionalLight(fillLightColor, fillLightIntensity);
    fillLight.position.set(-30, 50, -30);
    fillLight.target.position.set(0, 0, 0);
    fillLight.castShadow = false;
    scene.add(fillLight);
    scene.add(fillLight.target);

    // 4. Rim light (subtle backlight for better object definition)
    const rimLight = new THREE.DirectionalLight(rimLightColor, rimLightIntensity);
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


export function fitCameraToObject(camera: THREE.PerspectiveCamera, object: THREE.Mesh, offset = 1.25, controls?: ArcballControls) {
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

    // Optionally re-center controls
    if (controls) {
        controls.target.copy(center);
        controls.update();
    }
}

/**
 * Sync controls from source to target
 */
export function syncOrbitControls(
    sourceControls: ArcballControls,
    targetControls: ArcballControls
){
      const sourceCamera = sourceControls.object as THREE.PerspectiveCamera;
      sourceCamera.updateMatrix();
      // _gizmos is private in the class, but cloning its matrix is required to sync pan offsets
      (sourceControls as any)._gizmos?.updateMatrix();

      const cameraMatrix = sourceCamera.matrix.clone().elements.slice();
      const gizmoMatrix = ((sourceControls as any)._gizmos?.matrix as THREE.Matrix4 | undefined)?.clone().elements.slice();

      const arcballState = {
          arcballState: {
              cameraFar: sourceCamera.far,
              cameraNear: sourceCamera.near,
              cameraUp: sourceCamera.up.clone(),
              cameraZoom: sourceCamera.zoom,
              cameraMatrix: { elements: cameraMatrix },
              gizmoMatrix: { elements: gizmoMatrix ?? new THREE.Matrix4().identity().elements.slice() },
              target: sourceControls.target.toArray(),
          } as any,
      };

      if (sourceCamera.isPerspectiveCamera) {
          arcballState.arcballState.cameraFov = sourceCamera.fov;
      }

      targetControls.setStateFromJSON(JSON.stringify(arcballState));
}

/**
 * Alternative approach: Create scene with shared camera parameters
 * This creates a bidirectional sync where interaction on either renderer affects both
 * @param renderer1 First WebGLRenderer
 * @param renderer2 Second WebGLRenderer
 * @param config1 SceneConfig for first scene
 * @param config2 SceneConfig for second scene
 * @param syncSceneControls Whether to sync scene controls bidirectionally
 */
export function createBidirectionalSyncedScenes(
    renderer1: THREE.WebGLRenderer,
    renderer2: THREE.WebGLRenderer,
    config1: SceneConfig = {},
    config2: SceneConfig = {},
    syncSceneControls: boolean = false
): { sceneBundle1: SceneBundle; sceneBundle2: SceneBundle } {
    // Create both scenes
    const sceneBundle1 = createScene(renderer1, config1);
    const sceneBundle2 = createScene(renderer2, config2);

    if(!syncSceneControls){
        return { sceneBundle1, sceneBundle2 };
    }

    let isSyncing = false; // Prevent infinite loops
    const syncControls = (sourceControls: ArcballControls, targetControls: ArcballControls) => {
        if (isSyncing) return;
        isSyncing = true;
        syncOrbitControls(sourceControls, targetControls);
        isSyncing = false;
    }

    // Setup bidirectional sync
    const setupControlsSync = (controls1: ArcballControls, controls2: ArcballControls) => {
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
/**
 * Fits camera to view all meshes in the scene from the direction of z=0 plane
 * and stores the camera position for later reset functionality
 */
export function fitCameraToAllMeshesFromZPlane(sceneBundle: SceneBundle, offset: number = 1.25): void {
    const { scene, camera, controls } = sceneBundle;

    // Find all meshes in the scene
    const meshes: THREE.Mesh[] = [];
    scene.traverse((child) => {
        if (child.type === 'Mesh') {
            meshes.push(child as THREE.Mesh);
        }
    });

    if (meshes.length === 0) {
        console.warn('No meshes found in scene');
        return;
    }

    // Calculate bounding box of all meshes combined
    const combinedBox = new THREE.Box3();
    meshes.forEach(mesh => {
        const meshBox = new THREE.Box3().setFromObject(mesh);
        combinedBox.union(meshBox);
    });

    // Get the center and size of all meshes
    const center = combinedBox.getCenter(new THREE.Vector3());
    const size = combinedBox.getSize(new THREE.Vector3());

    // Calculate the maximum dimension to determine camera distance
    const maxDimension = Math.max(size.x, size.y, size.z);

    // Calculate distance needed to fit all objects in view
    // Account for aspect ratio - on portrait mobile, width is limiting factor
    const fov = camera.fov * (Math.PI / 180); // Convert to radians
    const fitHeightDistance = (maxDimension * offset) / (2 * Math.tan(fov / 2));
    const fitWidthDistance = fitHeightDistance / camera.aspect;
    const distance = Math.max(fitHeightDistance, fitWidthDistance);

    // Position camera to look at the scene from z=0 direction
    // This means the camera will be positioned along the positive Z axis
    const cameraPosition = new THREE.Vector3(center.x, center.y, center.z + distance);

    // Set camera position and orientation
    camera.position.copy(cameraPosition);
    camera.lookAt(center);

    // Update controls target to the center of all meshes
    controls.target.copy(center);
    controls.update();

    // Update camera projection matrix
    camera.updateProjectionMatrix();
}

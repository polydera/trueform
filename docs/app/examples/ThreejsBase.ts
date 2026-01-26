import type { MainModule } from "@/examples/native";
import * as THREE from "three";
import Stats from "stats-gl";
import {
  createBidirectionalSyncedScenes,
  createSceneWithCustomConfig,
  fitCameraToAllMeshesFromZPlane,
  type SceneBundle,
  syncOrbitControls,
} from "@/utils/sceneUtils";
import {
  initMeshGeometry,
  switchTextures,
  createInstancedMaterial,
} from "@/utils/utils";
import { ArcballControls } from 'three/addons/controls/ArcballControls.js';
import { SRGBColorSpace } from "three";

abstract class IThreejsBase {
  abstract runMain(): void;
  abstract getAverageTime(): number;
  abstract getAveragePickTime(): number;
  abstract updateMeshes(): void;
}

export abstract class ThreejsBase implements IThreejsBase {
  protected readonly wasmInstance: MainModule;
  protected paths: string[];

  // First renderer (primary scene)
  protected readonly renderer: THREE.WebGLRenderer;
  protected readonly sceneBundle1: SceneBundle;
  protected meshes = new Map<number, THREE.Mesh>(); // Legacy: individual meshes (unused with instancing)
  protected geometries = new Map<number, THREE.BufferGeometry>(); // Shared geometry per mesh_data_id
  // GPU Instancing: one InstancedMesh per unique mesh_data_id
  protected instancedMeshes = new Map<number, THREE.InstancedMesh>();
  // Maps instance_id -> { meshDataId, indexInBatch } for updating matrices/colors
  protected instanceIndices = new Map<number, { meshDataId: number; indexInBatch: number }>();
  protected stats = new Stats({ horizontal: false, trackGPU: true });
  protected isDarkMode: boolean;
  private showStats: boolean;
  private cleanupCallbacks: Array<() => void> = [];
  private animationFrameId: number | null = null;
  private disposed = false;
  private pointerDownListener?: (event: PointerEvent) => void;
  private pointerMoveListener?: (event: PointerEvent) => void;
  private pointerUpListener?: (event: PointerEvent) => void;
  private container: HTMLElement;
  private container2?: HTMLElement;

  protected readonly renderer2?: THREE.WebGLRenderer;
  protected readonly sceneBundle2?: SceneBundle;
  protected meshes2 = new Map<number, THREE.Mesh>();

  protected syncSceneControls = true;
  protected renderer2Interactive = false;
  protected lockRotationOnTouchDrag = false;

  private raycaster = new THREE.Raycaster();
  private ndc = new THREE.Vector2();
  private ray = new THREE.Ray();
  private touchDragPointerId: number | null = null;

  public refreshTimeValue?: () => number;

  constructor(
    wasmInstance: MainModule,
    paths: string[],
    container: HTMLElement,
    container2?: HTMLElement,
    skipUpdate?: boolean,
    showStats = true,
    isDarkMode = true,
  ) {
    this.container = container;
    this.container2 = container2;
    this.wasmInstance = wasmInstance;
    this.paths = paths;
    this.showStats = showStats;
    this.isDarkMode = isDarkMode;

    // Setup first renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.setClearColor(0x000000, 0.0);
    const rect = container.getBoundingClientRect();
    this.renderer.setSize(rect.width, rect.height);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.shadowMap.autoUpdate = true;
    container.innerHTML = "";
    container.appendChild(this.renderer.domElement);
    if (this.showStats) {
      this.stats.init(this.renderer);
      container.style.position = "relative";
      this.stats.dom.style.position = "absolute";
      container.appendChild(this.stats.dom);
    }

    // Setup second renderer if container2 is provided
    if (container2) {
      this.renderer2 = new THREE.WebGLRenderer({ antialias: true });
      this.renderer2.setPixelRatio(window.devicePixelRatio);
      this.renderer2.outputColorSpace = THREE.SRGBColorSpace;
      this.renderer2.toneMapping = THREE.ACESFilmicToneMapping;
      this.renderer2.toneMappingExposure = 1.0;
      const rect2 = container2.getBoundingClientRect();
      this.renderer2.setSize(rect2.width, rect2.height);
      this.renderer2.shadowMap.enabled = true;
      this.renderer2.shadowMap.type = THREE.PCFSoftShadowMap;
      this.renderer2.shadowMap.autoUpdate = true;
      container2.innerHTML = "";
      container2.appendChild(this.renderer2.domElement);
    }

    //////////////////////////// Scene Setup Using Utility Functions //////////////////////////////////////
    // Create synchronized scenes if we have both renderers
    if (this.renderer2) {
      const config1 = {
        backgroundColor: this.isDarkMode ? 0x262626 : 0xe5e5e5,
        cameraPosition: { x: 0, y: 50, z: 0 },
        cameraLookAt: { x: 0, y: 0, z: 0 },
        ambientLightIntensity: 0.8,
        directionalLightIntensity: 0.8,
        enableShadows: true,
      };
      const config2 = {
        backgroundColor: this.isDarkMode ? 0x404040 : 0xd1d1d1,
        cameraPosition: { x: 0, y: 50, z: 25 },
        cameraLookAt: { x: 0, y: 0, z: 0 },
        ambientLightIntensity: 0.8,
        directionalLightIntensity: 0.8,
        enableShadows: true,
      };

      // Use bidirectional synchronized scenes (interaction on either renderer affects both)
      const { sceneBundle1, sceneBundle2 } = createBidirectionalSyncedScenes(
        this.renderer,
        this.renderer2,
        config1,
        config2,
      );
      this.sceneBundle1 = sceneBundle1;
      this.sceneBundle2 = sceneBundle2;

      if (this.syncSceneControls) {
        let isSyncing = false; // Prevent infinite loops
        const syncControls = (
          sourceControls: ArcballControls,
          targetControls: ArcballControls,
        ) => {
          if (isSyncing) return;
          isSyncing = true;
          syncOrbitControls(sourceControls, targetControls);
          isSyncing = false;
        };
        const setupControlsSync = (controls1: ArcballControls, controls2: ArcballControls) => {
          const syncEvents = ["change", "start", "end"];

          syncEvents.forEach((eventType) => {
            controls1.addEventListener(eventType, () => {
              if (this.syncSceneControls) syncControls(controls1, controls2);
            });
          });
          controls2.addEventListener("start", () => {
            this.syncSceneControls = false;
          });
        };
        setupControlsSync(sceneBundle1.controls, sceneBundle2.controls);
        syncControls(sceneBundle1.controls, sceneBundle2.controls);
      }
    } else {
      // Create first scene with camera, controls, and lighting (single renderer mode)
      this.sceneBundle1 = createSceneWithCustomConfig(this.renderer, 1);
    }

    this.applyTheme(this.isDarkMode);
    this.animationFrameId = requestAnimationFrame(this.animate);

    // Setup ResizeObserver for container-specific resize handling
    const resizeObserver = new ResizeObserver((entries) => {
      if (this.disposed) return;

      for (const entry of entries) {
        const { width, height } = entry.contentRect;

        if (entry.target === container) {
          this.renderer.setSize(width, height);
          this.sceneBundle1.camera.aspect = width / height;
          this.sceneBundle1.camera.updateProjectionMatrix();
          this.sceneBundle1.controls.update();
          this.renderer.render(this.sceneBundle1.scene, this.sceneBundle1.camera);
        }

        if (entry.target === container2 && this.renderer2 && this.sceneBundle2) {
          this.renderer2.setSize(width, height);
          this.sceneBundle2.camera.aspect = width / height;
          this.sceneBundle2.camera.updateProjectionMatrix();
          this.sceneBundle2.controls.update();
          this.renderer2.render(this.sceneBundle2.scene, this.sceneBundle2.camera);
        }
      }
    });

    resizeObserver.observe(container);
    if (container2) {
      resizeObserver.observe(container2);
    }

    this.addCleanup(() => {
      resizeObserver.disconnect();
    });

    this.pointerDownListener = (event: PointerEvent) => {
      this.onPointerDown(event);
    };
    this.pointerMoveListener = (event: PointerEvent) => {
      this.onPointerMove(event);
    };
    this.pointerUpListener = (event: PointerEvent) => {
      this.onPointerUp(event);
    };

    this.renderer.domElement.addEventListener("pointerdown", this.pointerDownListener, true);
    this.renderer.domElement.addEventListener("pointermove", this.pointerMoveListener, true);
    this.renderer.domElement.addEventListener("pointerup", this.pointerUpListener, true);
    this.addCleanup(() => {
      if (this.pointerDownListener) {
        this.renderer.domElement.removeEventListener("pointerdown", this.pointerDownListener, true);
      }
      if (this.pointerMoveListener) {
        this.renderer.domElement.removeEventListener("pointermove", this.pointerMoveListener, true);
      }
      if (this.pointerUpListener) {
        this.renderer.domElement.removeEventListener("pointerup", this.pointerUpListener, true);
      }
    });

    // Add event listeners to second renderer if it exists
    if (this.renderer2 && this.renderer2Interactive) {
      this.renderer2.domElement.addEventListener("pointerdown", this.pointerDownListener, true);
      this.renderer2.domElement.addEventListener("pointermove", this.pointerMoveListener, true);
      this.renderer2.domElement.addEventListener("pointerup", this.pointerUpListener, true);
      this.addCleanup(() => {
        if (this.pointerDownListener) {
          this.renderer2?.domElement.removeEventListener(
            "pointerdown",
            this.pointerDownListener,
            true,
          );
        }
        if (this.pointerMoveListener) {
          this.renderer2?.domElement.removeEventListener(
            "pointermove",
            this.pointerMoveListener,
            true,
          );
        }
        if (this.pointerUpListener) {
          this.renderer2?.domElement.removeEventListener("pointerup", this.pointerUpListener, true);
        }
      });
    }

    this.runMain();

    // Build InstancedMesh objects grouped by mesh_data_id
    this.buildInstancedMeshes();

    if (!skipUpdate) {
      this.updateMeshes();
      fitCameraToAllMeshesFromZPlane(this.sceneBundle1);
      if (!this.syncSceneControls && this.sceneBundle2) {
        fitCameraToAllMeshesFromZPlane(this.sceneBundle2);
        syncOrbitControls(this.sceneBundle1.controls, this.sceneBundle2.controls);
      }
    }
  }
  public onPointerUp(event: PointerEvent) {
    const handled = this.wasmInstance.OnLeftButtonUp();
    this.updateMeshes();
    const isTouchDragLocked = this.isTouchDragLocked(event);
    if (isTouchDragLocked) {
      this.touchDragPointerId = null;
    }
    if (handled || isTouchDragLocked) {
      event.stopPropagation();
    }
  }
  public onPointerDown(event: PointerEvent) {
    let handled = false;
    if (event.pointerType == "touch") {
      const startedOnMesh = this.onPointerMove(event, true);
      if (this.lockRotationOnTouchDrag && startedOnMesh) {
        this.touchDragPointerId = event.pointerId;
      }
    }
    if (event.buttons === 1) handled = this.wasmInstance.OnLeftButtonDown();
    this.updateMeshes();
    if (handled || this.isTouchDragLocked(event)) {
      event.stopPropagation();
    }
  }
  public onPointerMove(event: PointerEvent, touchHover = false) {
    // Get bounding rect and mouse position
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.ndc.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.ndc.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Build world ray
    this.raycaster.setFromCamera(this.ndc, this.sceneBundle1.camera);
    this.ray.copy(this.raycaster.ray);
    // 2) reusable math objects
    const cameraPosition = this.sceneBundle1.camera.position.clone();
    const dir = new THREE.Vector3();
    this.sceneBundle1.camera.getWorldDirection(dir);
    const cameraFocalPoint = cameraPosition.clone().add(dir.multiplyScalar(100));
    let handled = false;
    if (event.buttons === 0 || event.buttons === 1) {
      // console.log("pointermove ray", cameraPosition, cameraFocalPoint, ray.origin, ray.direction)
      const v1 = this.ray.origin.clone();
      const v2 = this.ray.direction.clone();
      const v3 = cameraPosition;
      const v4 = cameraFocalPoint;
      handled = this.wasmInstance.OnMouseMove(
        [v1.x, v1.y, v1.z],
        [v2.x, v2.y, v2.z],
        [v3.x, v3.y, v3.z],
        [v4.x, v4.y, v4.z],
      );
    }
    this.updateMeshes();
    if ((handled && !touchHover) || this.isTouchDragLocked(event)) {
      event.stopPropagation();
    }
    return handled;
  }

  /**
   * Build InstancedMesh objects grouped by mesh_data_id.
   * Called once after runMain() when instances are created.
   */
  protected buildInstancedMeshes() {
    // Clear existing instanced meshes
    this.instancedMeshes.forEach((mesh) => {
      this.sceneBundle1.scene.remove(mesh);
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    });
    this.instancedMeshes.clear();
    this.instanceIndices.clear();
    this.geometries.clear();

    const numInstances = this.wasmInstance.get_number_of_instances();
    if (numInstances === 0) return;

    // Count instances per mesh_data_id
    const countPerMeshData = new Map<number, number>();
    for (let i = 0; i < numInstances; i++) {
      const inst = this.wasmInstance.get_instance_on_idx(i);
      if (!inst) continue;
      const meshDataId = inst.mesh_data_id;
      countPerMeshData.set(meshDataId, (countPerMeshData.get(meshDataId) ?? 0) + 1);
    }

    // Create InstancedMesh for each mesh_data_id
    const indexCounters = new Map<number, number>(); // tracks next index for each mesh_data_id

    for (const [meshDataId, count] of countPerMeshData) {
      // Get or create geometry
      if (!this.geometries.has(meshDataId)) {
        const meshData = this.wasmInstance.get_mesh_data_on_idx(meshDataId);
        if (meshData) {
          const geometry = new THREE.BufferGeometry();
          initMeshGeometry(meshData, { geometry } as THREE.Mesh);
          this.geometries.set(meshDataId, geometry);
        }
      }

      const geometry = this.geometries.get(meshDataId);
      if (!geometry) continue;

      // Create material and InstancedMesh
      const material = createInstancedMaterial(this.isDarkMode);
      const instancedMesh = new THREE.InstancedMesh(geometry, material, count);
      instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

      // Initialize instance colors (required for per-instance coloring)
      const colors = new Float32Array(count * 3);
      for (let j = 0; j < count * 3; j++) {
        colors[j] = 1.0; // Default white
      }
      instancedMesh.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);
      instancedMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);

      this.instancedMeshes.set(meshDataId, instancedMesh);
      this.sceneBundle1.scene.add(instancedMesh);
      indexCounters.set(meshDataId, 0);
    }

    // Build instanceIndices map
    for (let i = 0; i < numInstances; i++) {
      const inst = this.wasmInstance.get_instance_on_idx(i);
      if (!inst) continue;
      const meshDataId = inst.mesh_data_id;
      const indexInBatch = indexCounters.get(meshDataId) ?? 0;
      this.instanceIndices.set(i, { meshDataId, indexInBatch });
      indexCounters.set(meshDataId, indexInBatch + 1);
    }
  }

  public updateMeshes() {
    if (this.disposed) return;

    // Track which InstancedMesh needs update
    const needsMatrixUpdate = new Set<number>();
    const needsColorUpdate = new Set<number>();

    const tempMatrix = new THREE.Matrix4();
    const tempColor = new THREE.Color();

    for (let i = 0; i < this.wasmInstance.get_number_of_instances(); i++) {
      const inst = this.wasmInstance.get_instance_on_idx(i);
      const indices = this.instanceIndices.get(i);
      if (!inst || !indices) continue;

      const { meshDataId, indexInBatch } = indices;
      const instancedMesh = this.instancedMeshes.get(meshDataId);
      if (!instancedMesh) continue;

      // Update matrix
      if (inst.matrix_updated) {
        const matrix = new Float32Array(inst.get_matrix());
        tempMatrix.fromArray(matrix);
        tempMatrix.transpose();
        instancedMesh.setMatrixAt(indexInBatch, tempMatrix);
        needsMatrixUpdate.add(meshDataId);
      }

      // Update color
      const newC = inst.color;
      tempColor.setRGB(newC[0], newC[1], newC[2], SRGBColorSpace);
      instancedMesh.setColorAt(indexInBatch, tempColor);
      needsColorUpdate.add(meshDataId);
    }

    // Mark instance buffers as needing update
    for (const meshDataId of needsMatrixUpdate) {
      const mesh = this.instancedMeshes.get(meshDataId);
      if (mesh) mesh.instanceMatrix.needsUpdate = true;
    }
    for (const meshDataId of needsColorUpdate) {
      const mesh = this.instancedMeshes.get(meshDataId);
      if (mesh?.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
  }

  public getAverageTime() {
    return this.wasmInstance.get_average_time();
  }

  public getAveragePickTime() {
    return this.wasmInstance.get_average_pick_time();
  }

  abstract runMain(): void;

  public dispose() {
    if (this.disposed) return;
    this.disposed = true;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
    }
    this.cleanupCallbacks.forEach((cb) => cb());
    this.cleanupCallbacks = [];

    // Dispose instanced meshes
    this.instancedMeshes.forEach((mesh) => {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    });
    this.instancedMeshes.clear();
    this.instanceIndices.clear();
    this.geometries.clear();

    this.sceneBundle1.controls.dispose();
    this.disposeScene(this.sceneBundle1.scene);
    this.renderer.dispose();

    if (this.sceneBundle2 && this.renderer2) {
      this.sceneBundle2.controls.dispose();
      this.disposeScene(this.sceneBundle2.scene);
      this.renderer2.dispose();
    }

    if (this.showStats && this.stats?.dom) {
      this.stats.dom.remove();
    }

    this.container.innerHTML = "";
    if (this.container2) {
      this.container2.innerHTML = "";
    }
  }

  public isDisposed() {
    return this.disposed;
  }

  protected addCleanup(callback: () => void) {
    this.cleanupCallbacks.push(callback);
  }

  private isTouchDragLocked(event: PointerEvent) {
    return (
      this.lockRotationOnTouchDrag &&
      event.pointerType === "touch" &&
      this.touchDragPointerId !== null &&
      event.pointerId === this.touchDragPointerId
    );
  }

  private animate = () => {
    if (this.disposed) return;
    this.animationFrameId = requestAnimationFrame(this.animate);
    this.sceneBundle1.controls.update();
    this.renderer.render(this.sceneBundle1.scene, this.sceneBundle1.camera);
    if (this.renderer2 && this.sceneBundle2) {
      this.sceneBundle2.controls.update();
      this.renderer2.render(this.sceneBundle2.scene, this.sceneBundle2.camera);
    }
    if (this.refreshTimeValue) this.refreshTimeValue();
    if (this.showStats) {
      this.stats.update();
    }
  };

  private disposeScene(scene: THREE.Scene) {
    scene.traverse((object) => {
      const mesh = object as THREE.Mesh;
      if (mesh.geometry) {
        mesh.geometry.dispose();
      }
      if (Array.isArray(mesh.material)) {
        mesh.material.forEach((material) => material.dispose());
      } else if (mesh.material) {
        mesh.material.dispose();
      }
    });
  }

  private setSceneBackground(bundle: SceneBundle, renderer: THREE.WebGLRenderer, color: number) {
    bundle.scene.background = new THREE.Color(color);
    if (bundle.scene.fog) {
      bundle.scene.fog.color.set(color);
    }
    renderer.setClearColor(color, 1);
  }

  public applyTheme(isDark: boolean) {
    this.isDarkMode = isDark;
    const primaryBg = isDark ? 0x1e1e1e : 0xfafafa;
    const secondaryBg = isDark ? 0x262626 : 0xf5f5f5;

    this.setSceneBackground(this.sceneBundle1, this.renderer, primaryBg);
    if (this.renderer2 && this.sceneBundle2) {
      this.setSceneBackground(this.sceneBundle2, this.renderer2, secondaryBg);
    }

    // Update instanced mesh materials
    let path: string;
    if (isDark) {
      path =
        "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/635D52_A9BCC0_B1AEA0_819598.png";
    } else {
      path =
        "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/2D2D2F_C6C2C5_727176_94949B.png";
    }
    const newTexture = new THREE.TextureLoader().load(path);
    this.instancedMeshes.forEach((instancedMesh) => {
      const material = instancedMesh.material as THREE.MeshMatcapMaterial;
      if (material.matcap) {
        material.matcap.dispose();
      }
      material.matcap = newTexture;
    });

    // Legacy mesh support (for subclasses that may still use individual meshes)
    this.meshes.forEach((mesh) => {
      switchTextures(mesh, this.isDarkMode);
    });
    this.meshes2.forEach((mesh) => {
      switchTextures(mesh, this.isDarkMode);
    });
  }
}

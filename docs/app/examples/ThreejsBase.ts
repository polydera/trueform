import type { MainModule } from "@/examples/native";
import * as THREE from "three";
import Stats from "stats-gl";
import {
  createBidirectionalSyncedScenes,
  createSceneWithCustomConfig,
  fitCameraToAllMeshesFromZPlane,
  type SceneBundle,
} from "@/utils/sceneUtils";
import { createMesh, getMeshFromWasm } from "@/utils/utils";

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
  protected meshes = new Map<number, THREE.Mesh>();
  protected stats = new Stats({ horizontal: false, trackGPU: true });
  private isDarkMode: boolean;
  private showStats: boolean;
  private cleanupCallbacks: Array<() => void> = [];
  private animationFrameId: number | null = null;
  private disposed = false;
  private resizeListener?: () => void;
  private pointerDownListener?: (event: PointerEvent) => void;
  private pointerMoveListener?: (event: PointerEvent) => void;
  private pointerUpListener?: (event: PointerEvent) => void;
  private container: HTMLElement;
  private container2?: HTMLElement;

  protected readonly renderer2?: THREE.WebGLRenderer;
  protected readonly sceneBundle2?: SceneBundle;
  protected meshes2 = new Map<number, THREE.Mesh>();

  protected renderer2Interactive = false;

  private raycaster = new THREE.Raycaster();
  private ndc = new THREE.Vector2();
  private ray = new THREE.Ray();

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
    } else {
      // Create first scene with camera, controls, and lighting (single renderer mode)
      this.sceneBundle1 = createSceneWithCustomConfig(this.renderer, 1);
    }

    this.applyTheme(this.isDarkMode);
    this.animationFrameId = requestAnimationFrame(this.animate);

    // Add resize event listener
    this.resizeListener = () => {
      if (this.disposed) return;
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
    };
    window.addEventListener("resize", this.resizeListener);
    this.addCleanup(() => {
      if (this.resizeListener) {
        window.removeEventListener("resize", this.resizeListener);
      }
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

    for (let i = 0; i < this.wasmInstance.get_number_of_meshes(); i++) {
      const mesh = createMesh();
      this.meshes.set(i, mesh);
      this.sceneBundle1.scene.add(mesh);
    }

    if (!skipUpdate) {
      this.updateMeshes();
      fitCameraToAllMeshesFromZPlane(this.sceneBundle1);
    }
  }
  public onPointerUp(event: PointerEvent) {
    const handled = this.wasmInstance.OnLeftButtonUp();
    this.updateMeshes();
    if (handled) {
      event.stopPropagation();
    }
  }
  public onPointerDown(event: PointerEvent) {
    let handled = false;
    if (event.buttons === 1) handled = this.wasmInstance.OnLeftButtonDown();
    this.updateMeshes();
    if (handled) {
      event.stopPropagation();
    }
  }
  public onPointerMove(event: PointerEvent) {
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
    if (handled) {
      event.stopPropagation();
    }
  }

  public updateMeshes() {
    if (this.disposed) return;
    for (let i = 0; i < this.wasmInstance.get_number_of_meshes(); i++) {
      const wO = this.wasmInstance.get_mesh_on_idx(i);
      const mesh = this.meshes.get(i);
      if (!wO || !mesh) continue;
      getMeshFromWasm(wO, mesh);
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

    this.sceneBundle1.controls.dispose();
    this.disposeScene(this.sceneBundle1.scene);
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.renderer.domElement.remove();

    if (this.sceneBundle2 && this.renderer2) {
      this.sceneBundle2.controls.dispose();
      this.disposeScene(this.sceneBundle2.scene);
      this.renderer2.dispose();
      this.renderer2.forceContextLoss();
      this.renderer2.domElement.remove();
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

  private animate = () => {
    if (this.disposed) return;
    this.animationFrameId = requestAnimationFrame(this.animate);
    this.sceneBundle1.controls.update();
    this.renderer.render(this.sceneBundle1.scene, this.sceneBundle1.camera);
    if (this.renderer2 && this.sceneBundle2) {
      this.sceneBundle2.controls.update();
      this.renderer2.render(this.sceneBundle2.scene, this.sceneBundle2.camera);
    }
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
  }
}

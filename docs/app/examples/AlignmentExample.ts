import type { MainModule } from "@/examples/native";
import { ThreejsBase } from "@/examples/ThreejsBase";
import { fitCameraToAllMeshesFromZPlane } from "@/utils/sceneUtils";
import * as THREE from "three";

export type InteractionMode = "move" | "rotate";

export class AlignmentExample extends ThreejsBase {
  private alignmentTime = 0;
  private interactionMode: InteractionMode = "move";

  // Rotation state
  private isRotating = false;
  private lastMouseX = 0;
  private lastMouseY = 0;

  constructor(
    wasmInstance: MainModule,
    paths: string[],
    container: HTMLElement,
    isDarkMode = true,
  ) {
    // skipUpdate = true so we can position meshes before fitting camera
    super(wasmInstance, paths, container, undefined, true, false, isDarkMode);
    this.updateMeshes();
    this.positionMeshesForScreen(container);
    this.setupOrthographicCamera(container);

    // Warmup alignment (run once, then reposition)
    this.wasmInstance.alignment_run_align();
    this.positionMeshesForScreen(container);
    this.updateMeshes();
  }

  private setupOrthographicCamera(container: HTMLElement): void {
    const orthoCamera = new THREE.OrthographicCamera(
      -1, 1, 1, -1, 0.1, 1000
    );

    // Replace camera in scene bundle
    (this.sceneBundle1 as any).camera = orthoCamera;
    this.sceneBundle1.controls.setCamera(orthoCamera);

    // Now fit the orthographic camera to all meshes
    this.fitOrthographicCamera(container);
  }

  private fitOrthographicCamera(container: HTMLElement): void {
    const rect = container.getBoundingClientRect();
    const aspect = rect.width / rect.height;
    const isLandscape = rect.width > rect.height;
    const camera = this.sceneBundle1.camera as unknown as THREE.OrthographicCamera;

    // Get positions from both instances to compute bounding box
    const positions: THREE.Vector3[] = [];
    const diag = this.wasmInstance.alignment_get_aabb_diagonal() ?? 1;

    for (let i = 0; i < 2; i++) {
      const inst = this.wasmInstance.get_instance_on_idx(i);
      if (!inst) continue;
      const matrix = new Float32Array(inst.get_matrix());
      const m = new THREE.Matrix4().fromArray(matrix).transpose();
      const pos = new THREE.Vector3();
      pos.setFromMatrixPosition(m);
      positions.push(pos);
    }

    // Compute center and extent
    const center = new THREE.Vector3();
    if (positions.length >= 2) {
      center.addVectors(positions[0]!, positions[1]!).multiplyScalar(0.5);
    }

    const separation = positions.length >= 2 ? positions[0]!.distanceTo(positions[1]!) : 0;
    // Different zoom for landscape vs portrait
    const zoomFactor = isLandscape ? 0.5 : 0.7;
    const extent = (separation + diag) * zoomFactor;

    // Set orthographic frustum
    camera.left = -extent * aspect;
    camera.right = extent * aspect;
    camera.top = extent;
    camera.bottom = -extent;
    camera.updateProjectionMatrix();

    // Position camera
    camera.position.set(center.x, center.y, center.z + diag * 3);
    camera.lookAt(center);

    this.sceneBundle1.controls.target.copy(center);
    this.sceneBundle1.controls.update();
  }

  private positionMeshesForScreen(container: HTMLElement): void {
    const rect = container.getBoundingClientRect();
    const isLandscape = rect.width > rect.height;
    const diag = this.wasmInstance.alignment_get_aabb_diagonal() ?? 1;

    // More spacing for the axis we're spreading along
    // Landscape: spread in X, Portrait: spread in Z
    const spacing = isLandscape ? diag * 1.2 : diag * 1.0;

    // Target stays at origin, source gets offset
    // Camera on Z axis looking at XY plane: X = screen horizontal, Y = screen vertical
    // Landscape: target left, source right (positive X)
    // Portrait: target below, source above (positive Y)
    const offset = isLandscape
      ? [spacing, 0, 0]
      : [0, spacing, 0];

    // Build translation matrix for source
    const m = new THREE.Matrix4().makeTranslation(offset[0], offset[1], offset[2]);
    m.transpose();

    const arr = m.toArray() as [
      number, number, number, number,
      number, number, number, number,
      number, number, number, number,
      number, number, number, number
    ];
    this.wasmInstance.alignment_set_source_matrix(arr);
    this.updateMeshes();
  }

  public setMode(mode: InteractionMode): void {
    this.interactionMode = mode;
  }

  public getMode(): InteractionMode {
    return this.interactionMode;
  }

  // Override pointer handlers to support rotate mode
  public override onPointerDown(event: PointerEvent): void {
    if (this.interactionMode === "rotate" && event.buttons === 1) {
      // First do a mouse move to update selection state in WASM
      const rect = this.renderer.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(ndc, this.sceneBundle1.camera);
      const ray = raycaster.ray;
      const cameraPos = this.sceneBundle1.camera.position;
      const dir = new THREE.Vector3();
      this.sceneBundle1.camera.getWorldDirection(dir);
      const focalPoint = cameraPos.clone().add(dir.multiplyScalar(100));

      this.wasmInstance.OnMouseMove(
        [ray.origin.x, ray.origin.y, ray.origin.z],
        [ray.direction.x, ray.direction.y, ray.direction.z],
        [cameraPos.x, cameraPos.y, cameraPos.z],
        [focalPoint.x, focalPoint.y, focalPoint.z]
      );

      // Check if we hit a selectable mesh
      const hitMesh = this.wasmInstance.OnLeftButtonDown();
      if (hitMesh) {
        // Cancel WASM's drag mode, we handle rotation ourselves
        this.wasmInstance.OnLeftButtonUp();
        this.isRotating = true;
        this.lastMouseX = event.clientX;
        this.lastMouseY = event.clientY;
        this.sceneBundle1.controls.enabled = false;
        event.stopPropagation();
      }
    } else {
      super.onPointerDown(event);
    }
  }

  public override onPointerMove(event: PointerEvent, touchHover = false): boolean {
    if (this.interactionMode === "rotate" && this.isRotating) {
      const dx = event.clientX - this.lastMouseX;
      const dy = event.clientY - this.lastMouseY;
      this.lastMouseX = event.clientX;
      this.lastMouseY = event.clientY;

      // Convert mouse movement to rotation (similar to C++ example)
      const angleX = dy * 0.5; // degrees
      const angleY = dx * 0.5;

      // Get current source matrix (source is instance 1)
      const sourceInst = this.wasmInstance.get_instance_on_idx(1);
      if (!sourceInst) return true;

      const matrix = new Float32Array(sourceInst.get_matrix());
      const m = new THREE.Matrix4().fromArray(matrix).transpose();

      // Get rotation center (translation part of current matrix)
      const center = new THREE.Vector3();
      center.setFromMatrixPosition(m);

      // Create rotation matrices around world X and Y axes
      const rotX = new THREE.Matrix4().makeRotationAxis(
        new THREE.Vector3(1, 0, 0),
        THREE.MathUtils.degToRad(angleX)
      );
      const rotY = new THREE.Matrix4().makeRotationAxis(
        new THREE.Vector3(0, 1, 0),
        THREE.MathUtils.degToRad(angleY)
      );

      // Apply rotations centered at the mesh position
      const toOrigin = new THREE.Matrix4().makeTranslation(-center.x, -center.y, -center.z);
      const fromOrigin = new THREE.Matrix4().makeTranslation(center.x, center.y, center.z);

      const newMatrix = new THREE.Matrix4()
        .multiply(fromOrigin)
        .multiply(rotY)
        .multiply(rotX)
        .multiply(toOrigin)
        .multiply(m);

      // Send to WASM
      newMatrix.transpose();
      const arr = newMatrix.toArray() as [
        number, number, number, number,
        number, number, number, number,
        number, number, number, number,
        number, number, number, number
      ];
      this.wasmInstance.alignment_set_source_matrix(arr);
      this.updateMeshes();

      event.stopPropagation();
      return true;
    } else {
      return super.onPointerMove(event, touchHover);
    }
  }

  public override onPointerUp(event: PointerEvent): void {
    if (this.isRotating) {
      this.isRotating = false;
      this.sceneBundle1.controls.enabled = true;
      event.stopPropagation();
    } else {
      super.onPointerUp(event);
    }
  }

  public runMain() {
    const v = new this.wasmInstance.VectorString();
    for (const path of this.paths) {
      v.push_back(path);
    }
    this.wasmInstance.run_main_alignment(v);
    for (const path of this.paths) {
      this.wasmInstance.FS.unlink(path);
    }
  }

  public align(): number {
    this.alignmentTime = this.wasmInstance.alignment_run_align();
    this.updateMeshes();
    return this.alignmentTime;
  }

  public isAligned(): boolean {
    return this.wasmInstance.alignment_is_aligned();
  }

  public getAlignmentTime(): number {
    return this.alignmentTime;
  }
}

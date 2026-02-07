import type { MainModule } from "@/examples/native";
import { ThreejsBase } from "@/examples/ThreejsBase";
import * as THREE from "three";

export class PositioningExample extends ThreejsBase {
  private keyPressed = false;

  // Closest points visualization
  private closestPointsGroup!: THREE.Group;
  private sphere1!: THREE.Mesh;
  private sphere2!: THREE.Mesh;
  private connector!: THREE.Mesh;

  constructor(
    wasmInstance: MainModule,
    paths: string[],
    container: HTMLElement,
    isDarkMode = true,
  ) {
    // skipUpdate = true so we can set up camera before fitting
    super(wasmInstance, paths, container, undefined, true, false, isDarkMode);
    this.updateMeshes();

    const interceptKeyDownEvent = (event: KeyboardEvent) => {
      if (this.keyPressed) return;
      this.keyPressed = true;
      this.wasmInstance.OnKeyPress(event.key);
      this.updateMeshes();
    };
    const interceptKeyUpEvent = (_event: KeyboardEvent) => {
      this.keyPressed = false;
    };
    window.addEventListener("keydown", interceptKeyDownEvent);
    window.addEventListener("keyup", interceptKeyUpEvent);

    this.setupClosestPointsVisuals();
    this.positionMeshesForScreen(container);
    this.setupOrthographicCamera(container);

    // Update to show initial visualization
    this.updateMeshes();
  }

  private setupOrthographicCamera(container: HTMLElement): void {
    const orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 1000);

    // Replace camera in scene bundle
    (this.sceneBundle1 as unknown as { camera: THREE.Camera }).camera = orthoCamera;
    this.sceneBundle1.controls.setCamera(orthoCamera);

    this.fitOrthographicCamera(container);
  }

  private positionMeshesForScreen(container: HTMLElement): void {
    const rect = container.getBoundingClientRect();
    const isLandscape = rect.width > rect.height;
    const aabbDiag = this.wasmInstance.positioning_get_aabb_diagonal?.() ?? 10;

    // Spacing between meshes
    const spacing = isLandscape ? aabbDiag * 1.2 : aabbDiag * 1.0;

    // Camera on Z axis looking at XY plane: X = screen horizontal, Y = screen vertical
    // Landscape: side by side (X axis)
    // Portrait: stacked (Y axis)
    const offsets: [number, number, number][] = isLandscape
      ? [[-spacing / 2, 0, 0], [spacing / 2, 0, 0]]
      : [[0, -spacing / 2, 0], [0, spacing / 2, 0]];

    for (let i = 0; i < 2; i++) {
      const [ox, oy, oz] = offsets[i]!;
      const m = new THREE.Matrix4().makeTranslation(ox, oy, oz);
      m.transpose();

      const arr = m.toArray() as [
        number, number, number, number,
        number, number, number, number,
        number, number, number, number,
        number, number, number, number
      ];
      this.wasmInstance.positioning_set_instance_matrix?.(i, arr);
    }
    this.updateMeshes();
  }

  private fitOrthographicCamera(container: HTMLElement): void {
    const rect = container.getBoundingClientRect();
    const aspect = rect.width / rect.height;
    const isLandscape = rect.width > rect.height;
    const camera = this.sceneBundle1.camera as unknown as THREE.OrthographicCamera;

    // Get bounding box of all instances
    const positions: THREE.Vector3[] = [];
    const aabbDiag = this.wasmInstance.positioning_get_aabb_diagonal?.() ?? 10;

    for (let i = 0; i < this.wasmInstance.get_number_of_instances(); i++) {
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
    const extent = (separation + aabbDiag) * zoomFactor;

    // Set orthographic frustum
    camera.left = -extent * aspect;
    camera.right = extent * aspect;
    camera.top = extent;
    camera.bottom = -extent;
    camera.updateProjectionMatrix();

    // Position camera
    camera.position.set(center.x, center.y, center.z + aabbDiag * 3);
    camera.lookAt(center);

    this.sceneBundle1.controls.target.copy(center);
    this.sceneBundle1.controls.update();
  }

  private setupClosestPointsVisuals(): void {
    this.closestPointsGroup = new THREE.Group();
    this.closestPointsGroup.name = "closest_points";

    // Sphere geometry and materials
    const sphereGeom = new THREE.SphereGeometry(1, 16, 12);
    const sphereMat = new THREE.MeshStandardMaterial({
      color: 0x00d5be, // Bright teal (like cross-section boundary)
      roughness: 0.3,
      metalness: 0.1,
    });

    // Cylinder geometry and material
    const cylGeom = new THREE.CylinderGeometry(1, 1, 1, 8);
    const cylMat = new THREE.MeshStandardMaterial({
      color: 0x00a89a, // Darker teal (like cross-section fill)
      roughness: 0.3,
      metalness: 0.1,
    });

    this.sphere1 = new THREE.Mesh(sphereGeom, sphereMat);
    this.sphere2 = new THREE.Mesh(sphereGeom.clone(), sphereMat.clone());
    this.connector = new THREE.Mesh(cylGeom, cylMat);

    this.closestPointsGroup.add(this.sphere1);
    this.closestPointsGroup.add(this.sphere2);
    this.closestPointsGroup.add(this.connector);

    this.closestPointsGroup.visible = false;
    this.sceneBundle1.scene.add(this.closestPointsGroup);
  }

  private updateClosestPointsVisuals(): void {
    if (!this.closestPointsGroup) return;

    const hasPoints = this.wasmInstance.positioning_has_closest_points?.();
    if (!hasPoints) {
      this.closestPointsGroup.visible = false;
      return;
    }

    const pts = this.wasmInstance.positioning_get_closest_points?.() as number[] | undefined;
    if (!pts || pts.length < 6) {
      this.closestPointsGroup.visible = false;
      return;
    }

    const p0 = new THREE.Vector3(pts[0], pts[1], pts[2]);
    const p1 = new THREE.Vector3(pts[3], pts[4], pts[5]);
    const dist = p0.distanceTo(p1);

    // Fixed size based on AABB diagonal (1.5% for spheres, 0.75% for cylinder)
    const aabbDiag = this.wasmInstance.positioning_get_aabb_diagonal?.() ?? 10;
    const sphereRadius = aabbDiag * 0.015;
    const cylRadius = aabbDiag * 0.0075;

    // Position and scale spheres
    this.sphere1.position.copy(p0);
    this.sphere1.scale.setScalar(sphereRadius);

    this.sphere2.position.copy(p1);
    this.sphere2.scale.setScalar(sphereRadius);

    // Position and orient cylinder
    if (dist > 0.001) {
      const mid = new THREE.Vector3().addVectors(p0, p1).multiplyScalar(0.5);
      this.connector.position.copy(mid);

      // Orient cylinder to point from p0 to p1
      const direction = new THREE.Vector3().subVectors(p1, p0).normalize();
      const quaternion = new THREE.Quaternion();
      quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
      this.connector.quaternion.copy(quaternion);

      // Scale: radius for X/Z, length for Y
      this.connector.scale.set(cylRadius, dist, cylRadius);
      this.connector.visible = true;
    } else {
      this.connector.visible = false;
    }

    this.closestPointsGroup.visible = true;
  }

  public override updateMeshes(): void {
    super.updateMeshes();
    this.updateClosestPointsVisuals();
    this.refreshTimeValue?.();
  }

  public override getAverageTime(): number {
    return this.wasmInstance.get_average_time();
  }

  public randomize() {
    this.wasmInstance.OnKeyPress("n");
    this.updateMeshes();
  }

  public override onPointerUp(event: PointerEvent) {
    const cameraPosition = this.sceneBundle1.camera.position.clone();
    const dir = new THREE.Vector3();
    this.sceneBundle1.camera.getWorldDirection(dir);
    const cameraFocalPoint = cameraPosition.clone().add(dir.multiplyScalar(100));

    const updateFocalPoint = (x: number, y: number, z: number) => {
      this.sceneBundle1.controls.target.set(x, y, z);
      this.sceneBundle1.controls.update();

      // Update instanced mesh matrices
      const tempMatrix = new THREE.Matrix4();
      const needsUpdate = new Set<number>();

      for (let i = 0; i < this.wasmInstance.get_number_of_instances(); i++) {
        const inst = this.wasmInstance.get_instance_on_idx(i);
        const indices = this.instanceIndices.get(i);
        if (!inst || !indices) continue;

        const { meshDataId, indexInBatch } = indices;
        const instancedMesh = this.instancedMeshes.get(meshDataId);
        if (!instancedMesh) continue;

        const matrix = new Float32Array(inst.get_matrix());
        tempMatrix.fromArray(matrix);
        tempMatrix.transpose();
        instancedMesh.setMatrixAt(indexInBatch, tempMatrix);
        needsUpdate.add(meshDataId);
      }

      for (const meshDataId of needsUpdate) {
        const mesh = this.instancedMeshes.get(meshDataId);
        if (mesh) mesh.instanceMatrix.needsUpdate = true;
      }

      this.sceneBundle1.scene.updateMatrixWorld(true);
      this.renderer.render(this.sceneBundle1.scene, this.sceneBundle1.camera);
    };

    let t = 0;
    const stepPositioning = () => {
      if (this.isDisposed()) return;
      t = this.wasmInstance.OnLeftButtonUpCustom(
        [cameraFocalPoint.x, cameraFocalPoint.y, cameraFocalPoint.z],
        updateFocalPoint,
        t,
      );
      if (t < 1.0) {
        requestAnimationFrame(stepPositioning);
      }
      this.updateMeshes();
      if (t < 1) {
        event.stopPropagation();
      }
    };
    requestAnimationFrame(stepPositioning);
  }

  public runMain() {
    const v = new this.wasmInstance.VectorString();
    for (const path of this.paths) {
      v.push_back(path);
    }
    this.wasmInstance.run_main_positioning(v);
    for (const path of this.paths) {
      this.wasmInstance.FS.unlink(path);
    }
  }
}

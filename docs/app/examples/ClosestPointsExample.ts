import { fitCameraToAllMeshesFromZPlane, createScene, type SceneBundle } from "@/utils/sceneUtils";
import { centerAndScale, pickMesh, randomTransformation, RollingAverage } from "@/utils/utils";
import * as THREE from "three";

type TF = typeof import("@/examples/trueform/index.js");

export class ClosestPointsExample {
  private tf: TF;
  private baseMesh: any;
  private tfMeshes: any[] = [];
  private container: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private sceneBundle: SceneBundle;
  private threeMeshes: THREE.Mesh[] = [];
  private materials: THREE.MeshMatcapMaterial[] = [];
  private running = true;
  private cleanups: (() => void)[] = [];

  // Interaction
  private selectedId: number | null = null;
  private dragging = false;
  private movingPlane = new THREE.Plane();
  private lastPoint = new THREE.Vector3();
  private raycaster = new THREE.Raycaster();
  private ndc = new THREE.Vector2();

  // Closest points visualization
  private closestGroup!: THREE.Group;
  private sphere1!: THREE.Mesh;
  private sphere2!: THREE.Mesh;
  private connector!: THREE.Mesh;
  private hasClosestPoints = false;
  private closestPtSelected = new THREE.Vector3(); // on the dragged mesh
  private closestPtOther = new THREE.Vector3();    // on the other mesh
  private aabbDiag = 10;

  // Animation (pointer-up slide)
  private animating = false;
  private animSelectedId = 0;
  private animRay: { origin: THREE.Vector3; dir: THREE.Vector3 } | null = null;
  private animFocalRay: { origin: THREE.Vector3; dir: THREE.Vector3 } | null = null;
  private animPrev = new THREE.Vector3();
  private animStart = 0;

  // Timing
  private timing = new RollingAverage();

  public refreshTimeValue: (() => number) | null = null;

  constructor(tf: TF, fileBuffer: ArrayBuffer, fileName: string, container: HTMLElement, isDarkMode = true) {
    this.tf = tf;
    this.container = container;

    // Load mesh
    const ext = fileName.split(".").pop()?.toLowerCase();
    this.baseMesh = ext === "stl" ? tf.readStl(fileBuffer) : tf.readObj(fileBuffer);
    centerAndScale(tf, this.baseMesh);

    // Compute AABB diagonal for sizing
    const points = this.baseMesh.points;
    const pMin = tf.min(points, 0);
    const pMax = tf.max(points, 0);
    const diag = pMax.sub(pMin);
    this.aabbDiag = tf.norm(diag) as number;
    pMin.delete(); pMax.delete(); diag.delete();

    // Create 2 tf meshes, positioned based on screen orientation
    const rect = container.getBoundingClientRect();
    const isLandscape = rect.width > rect.height;
    const spacing = isLandscape ? this.aabbDiag * 1.2 : this.aabbDiag * 1.0;
    const offsets: [number, number, number][] = isLandscape
      ? [[-spacing / 2, 0, 0], [spacing / 2, 0, 0]]
      : [[0, -spacing / 2, 0], [0, spacing / 2, 0]];

    const mesh0 = this.baseMesh;
    const mesh1 = this.baseMesh.shallowCopy();
    // No random rotation on first frame — just translate into position
    const mat0 = tf.makeTranslation(...offsets[0]!);
    mesh0.transformation = mat0;
    mat0.delete();
    const mat1 = tf.makeTranslation(...offsets[1]!);
    mesh1.transformation = mat1;
    mat1.delete();
    this.tfMeshes = [mesh0, mesh1];

    // Three.js setup
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    container.appendChild(this.renderer.domElement);

    this.sceneBundle = createScene(this.renderer, {
      backgroundColor: isDarkMode ? 0x1e1e1e : 0xfafafa,
      enableFog: false,
    });

    // Shared geometry from base mesh
    const faces = this.baseMesh.faces;
    const sharedGeometry = new THREE.BufferGeometry();
    sharedGeometry.setAttribute("position", new THREE.BufferAttribute(points.data, 3));
    sharedGeometry.setIndex(new THREE.BufferAttribute(
      new Uint32Array(faces.data.buffer, faces.data.byteOffset, faces.data.length), 1,
    ));
    sharedGeometry.computeVertexNormals();
    sharedGeometry.computeBoundingSphere();

    // Matcap URL
    const matcapUrl = isDarkMode
      ? "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/635D52_A9BCC0_B1AEA0_819598.png"
      : "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/2D2D2F_C6C2C5_727176_94949B.png";

    // Create 2 Three.js meshes
    for (let idx = 0; idx < 2; idx++) {
      const material = new THREE.MeshMatcapMaterial({
        side: THREE.DoubleSide,
        flatShading: true,
        color: new THREE.Color(0.8, 0.8, 0.8),
      });
      this.materials.push(material);
      const mesh = new THREE.Mesh(sharedGeometry, material);
      mesh.matrixAutoUpdate = false;
      this.syncThreeMatrix(idx, mesh);
      this.sceneBundle.scene.add(mesh);
      this.threeMeshes.push(mesh);
    }

    // Load matcap texture
    new THREE.TextureLoader().load(matcapUrl, (tex) => {
      for (let i = 0; i < this.materials.length; i++) {
        this.materials[i].matcap = i === 0 ? tex : tex.clone();
        this.materials[i].needsUpdate = true;
      }
    });

    // Closest points visuals
    this.setupClosestPointsVisuals();

    // Compute initial closest points
    this.computeClosestPoints();

    // Fit camera
    fitCameraToAllMeshesFromZPlane(this.sceneBundle, 1.5);

    // Pointer events
    const onPointerMove = (e: PointerEvent) => {
      if (!container.contains(e.target as Node)) return;
      this.updateNDC(e);
      if (this.dragging && this.selectedId !== null) {
        this.handleDrag();
      } else if (!this.dragging && !this.animating) {
        this.handleHover();
      }
    };
    const onPointerDown = (e: PointerEvent) => {
      if (!container.contains(e.target as Node)) return;
      if (this.animating) return;
      this.updateNDC(e);
      this.handleHover();
      if (this.selectedId !== null) {
        this.dragging = true;
        this.sceneBundle.controls.enabled = false;
      }
    };
    const onPointerUp = () => {
      if (this.animating) return;
      if (this.dragging && this.selectedId !== null && this.hasClosestPoints) {
        this.startSnapAnimation();
      } else {
        this.dragging = false;
        this.selectedId = null;
        this.sceneBundle.controls.enabled = true;
      }
    };

    container.addEventListener("pointermove", onPointerMove);
    container.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointerup", onPointerUp);
    this.cleanups.push(() => {
      container.removeEventListener("pointermove", onPointerMove);
      container.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointerup", onPointerUp);
    });

    // Resize
    const resizeObs = new ResizeObserver(() => this.resize());
    resizeObs.observe(container);
    this.cleanups.push(() => resizeObs.disconnect());
    this.resize();

    // Animate
    this.animate();
  }

  // ========== Closest points visuals ==========

  private setupClosestPointsVisuals() {
    this.closestGroup = new THREE.Group();

    const sphereGeom = new THREE.SphereGeometry(1, 16, 12);
    const sphereMat = new THREE.MeshStandardMaterial({
      color: 0x00d5be,
      roughness: 0.3,
      metalness: 0.1,
    });
    const cylGeom = new THREE.CylinderGeometry(1, 1, 1, 8);
    const cylMat = new THREE.MeshStandardMaterial({
      color: 0x00a89a,
      roughness: 0.3,
      metalness: 0.1,
    });

    this.sphere1 = new THREE.Mesh(sphereGeom, sphereMat);
    this.sphere2 = new THREE.Mesh(sphereGeom.clone(), sphereMat.clone());
    this.connector = new THREE.Mesh(cylGeom, cylMat);

    this.closestGroup.add(this.sphere1);
    this.closestGroup.add(this.sphere2);
    this.closestGroup.add(this.connector);
    this.closestGroup.visible = false;
    this.sceneBundle.scene.add(this.closestGroup);
  }

  private updateClosestVisuals() {
    if (!this.hasClosestPoints) {
      this.closestGroup.visible = false;
      return;
    }

    const sphereRadius = this.aabbDiag * 0.015;
    const cylRadius = this.aabbDiag * 0.0075;
    const dist = this.closestPtSelected.distanceTo(this.closestPtOther);

    this.sphere1.position.copy(this.closestPtSelected);
    this.sphere1.scale.setScalar(sphereRadius);

    this.sphere2.position.copy(this.closestPtOther);
    this.sphere2.scale.setScalar(sphereRadius);

    if (dist > 0.001) {
      const mid = new THREE.Vector3().addVectors(this.closestPtSelected, this.closestPtOther).multiplyScalar(0.5);
      this.connector.position.copy(mid);
      const direction = new THREE.Vector3().subVectors(this.closestPtOther, this.closestPtSelected).normalize();
      const quaternion = new THREE.Quaternion();
      quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
      this.connector.quaternion.copy(quaternion);
      this.connector.scale.set(cylRadius, dist, cylRadius);
      this.connector.visible = true;
    } else {
      this.connector.visible = false;
    }

    this.closestGroup.visible = true;
  }

  // ========== Closest points computation ==========

  private computeClosestPoints(draggedId = 0) {
    const tf = this.tf;
    const otherId = draggedId === 0 ? 1 : 0;
    const t0 = performance.now();
    if (tf.intersects(this.tfMeshes[draggedId], this.tfMeshes[otherId])) {
      this.hasClosestPoints = false;
    } else {
      // neighborSearch(dragged, other) → point0 on dragged, point1 on other
      const result = tf.neighborSearch(this.tfMeshes[draggedId], this.tfMeshes[otherId]);
      this.closestPtSelected.fromArray(result.point0.data as Float32Array);
      this.closestPtOther.fromArray(result.point1.data as Float32Array);
      result.point0.delete();
      result.point1.delete();
      this.hasClosestPoints = true;
    }
    this.timing.add(performance.now() - t0);
    this.updateClosestVisuals();
    if (this.refreshTimeValue) this.refreshTimeValue();
  }

  // ========== Pointer interaction ==========

  private updateNDC(e: PointerEvent) {
    const rect = this.container.getBoundingClientRect();
    this.ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  }

  private handleHover() {
    const tf = this.tf;
    this.raycaster.setFromCamera(this.ndc, this.sceneBundle.camera);
    const o = this.raycaster.ray.origin;
    const d = this.raycaster.ray.direction;
    const tfRay = tf.ray([o.x, o.y, o.z, d.x, d.y, d.z]);

    const hit = pickMesh(tf, tfRay, this.tfMeshes);
    tfRay.delete();

    if (hit) {
      this.selectedId = hit.index;
      const hitPoint = this.raycaster.ray.at(hit.t, new THREE.Vector3());
      const camDir = new THREE.Vector3();
      this.sceneBundle.camera.getWorldDirection(camDir);
      this.movingPlane.setFromNormalAndCoplanarPoint(camDir, hitPoint);
      this.lastPoint.copy(hitPoint);
    } else {
      this.selectedId = null;
    }
  }

  private handleDrag() {
    const id = this.selectedId!;

    this.raycaster.setFromCamera(this.ndc, this.sceneBundle.camera);
    const nextPoint = new THREE.Vector3();
    this.raycaster.ray.intersectPlane(this.movingPlane, nextPoint);
    if (!nextPoint) return;

    const dx = nextPoint.x - this.lastPoint.x;
    const dy = nextPoint.y - this.lastPoint.y;
    const dz = nextPoint.z - this.lastPoint.z;
    this.lastPoint.copy(nextPoint);

    // Update tf mesh transformation
    const mat = this.tfMeshes[id].transformation;
    const d = mat.data;
    d[3] += dx; d[7] += dy; d[11] += dz;
    this.tfMeshes[id].transformation = mat;
    mat.delete();

    this.syncThreeMatrix(id);
    this.computeClosestPoints(id);
  }

  // ========== Snap animation ==========

  private startSnapAnimation() {
    this.animSelectedId = this.selectedId!;
    const origin = this.closestPtSelected.clone();
    const dir = new THREE.Vector3().subVectors(this.closestPtOther, this.closestPtSelected);
    this.animRay = { origin, dir };

    // Animate camera focal point toward the other mesh's closest point
    const camTarget = (this.sceneBundle.controls as any).target.clone();
    this.animFocalRay = {
      origin: camTarget,
      dir: new THREE.Vector3().subVectors(this.closestPtOther, camTarget),
    };

    this.animPrev.copy(origin);
    this.animStart = performance.now();
    this.animating = true;
    this.dragging = false;
    this.selectedId = null;
    this.sceneBundle.controls.enabled = false;
  }

  private stepAnimation() {
    if (!this.animRay) return;

    const elapsed = (performance.now() - this.animStart) / 1000;
    const t = Math.min(1, 1.75 * elapsed);
    const s = 3 * t * t - 2 * t * t * t; // smooth-step

    const pt = this.animRay.origin.clone().addScaledVector(this.animRay.dir, s);
    const delta = pt.clone().sub(this.animPrev);

    // Move tf mesh
    const mat = this.tfMeshes[this.animSelectedId].transformation;
    const d = mat.data;
    d[3] += delta.x; d[7] += delta.y; d[11] += delta.z;
    this.tfMeshes[this.animSelectedId].transformation = mat;
    mat.delete();
    this.syncThreeMatrix(this.animSelectedId);
    this.animPrev.copy(pt);

    // Update closest point visualization (shrinking line)
    this.closestPtSelected.copy(pt);
    this.updateClosestVisuals();

    // Animate focal point
    if (this.animFocalRay) {
      const focal = this.animFocalRay.origin.clone().addScaledVector(this.animFocalRay.dir, s);
      (this.sceneBundle.controls as any).target.copy(focal);
    }

    if (t >= 1) {
      this.animating = false;
      this.animRay = null;
      this.animFocalRay = null;
      this.hasClosestPoints = false;
      this.updateClosestVisuals();
      this.sceneBundle.controls.enabled = true;
    }
  }

  // ========== Helpers ==========

  private syncThreeMatrix(index: number, mesh?: THREE.Mesh) {
    const target = mesh ?? this.threeMeshes[index];
    const mat = this.tfMeshes[index].transformation;
    if (!mat) return;
    const m = new THREE.Matrix4();
    m.fromArray(mat.data);
    m.transpose();
    target.matrix.copy(m);
    mat.delete();
  }

  // ========== Public API ==========

  public getAverageTime(): number {
    return this.timing.average;
  }

  public randomize() {
    const tf = this.tf;
    for (let i = 0; i < this.tfMeshes.length; i++) {
      // Keep current translation, re-roll rotation
      const oldMat = this.tfMeshes[i].transformation;
      const tx = oldMat.data[3];
      const ty = oldMat.data[7];
      const tz = oldMat.data[11];
      oldMat.delete();

      const newMat = randomTransformation(tf, tx, ty, tz);
      this.tfMeshes[i].transformation = newMat;
      newMat.delete();
      this.syncThreeMatrix(i);
    }
    this.computeClosestPoints();
  }

  public applyTheme(isDark: boolean) {
    this.sceneBundle.scene.background = new THREE.Color(isDark ? 0x1e1e1e : 0xfafafa);
    const matcapUrl = isDark
      ? "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/635D52_A9BCC0_B1AEA0_819598.png"
      : "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/2D2D2F_C6C2C5_727176_94949B.png";
    new THREE.TextureLoader().load(matcapUrl, (tex) => {
      for (let i = 0; i < this.materials.length; i++) {
        if (this.materials[i].matcap) this.materials[i].matcap!.dispose();
        this.materials[i].matcap = i === 0 ? tex : tex.clone();
        this.materials[i].needsUpdate = true;
      }
    });
  }

  private resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.sceneBundle.camera.aspect = w / h;
    this.sceneBundle.camera.updateProjectionMatrix();
  }

  private animate() {
    if (!this.running) return;
    requestAnimationFrame(() => this.animate());
    if (this.animating) {
      this.stepAnimation();
    }
    this.sceneBundle.controls.update();
    this.renderer.render(this.sceneBundle.scene, this.sceneBundle.camera);
  }

  public dispose() {
    this.running = false;
    for (const fn of this.cleanups) fn();
    this.cleanups = [];
    this.sceneBundle.controls.dispose();

    // Dispose shared geometry (only once)
    if (this.threeMeshes.length > 0) {
      this.threeMeshes[0].geometry.dispose();
    }
    for (const mesh of this.threeMeshes) {
      (mesh.material as THREE.Material).dispose();
    }

    // Dispose closest points visuals
    this.sphere1.geometry.dispose();
    (this.sphere1.material as THREE.Material).dispose();
    this.sphere2.geometry.dispose();
    (this.sphere2.material as THREE.Material).dispose();
    this.connector.geometry.dispose();
    (this.connector.material as THREE.Material).dispose();

    this.renderer.dispose();
    this.renderer.domElement.remove();

    // Delete tf meshes (shallow copy first, then baseMesh)
    for (let i = this.tfMeshes.length - 1; i >= 0; i--) {
      this.tfMeshes[i].delete();
    }
    this.tfMeshes = [];
  }
}

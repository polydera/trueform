import { createScene, type SceneBundle } from "@/utils/sceneUtils";
import { centerAndScale, pickMesh } from "@/utils/utils";
import * as THREE from "three";
import { ArcballControls } from "three/addons/controls/ArcballControls.js";

type TF = typeof import("@polydera/trueform");

export type InteractionMode = "move" | "rotate";

// ════════════════════════════════════════════════════════════════════════════
// Matrix convention helpers: TF row-major ↔ Three.js column-major
// ════════════════════════════════════════════════════════════════════════════

function tfToThree(tfObj: any): THREE.Matrix4 {
  const mat = tfObj.transformation;
  if (!mat) return new THREE.Matrix4();
  const m = new THREE.Matrix4().fromArray(mat.data).transpose();
  mat.delete();
  return m;
}

function threeToTf(tf: TF, m: THREE.Matrix4): any {
  const rm = m.clone().transpose();
  const mat = tf.ndarray(new Float32Array(rm.toArray())).reshape([4, 4]);
  return mat; // caller must set .transformation then .delete()
}

export class AlignmentExample {
  private tf: TF;
  private smoothedTarget: any;
  private source: any;
  private targetPC: any;
  private sourcePC: any;
  private aabbDiagonal: number;

  private container: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private sceneBundle: SceneBundle;

  private targetThree: THREE.Mesh;
  private sourceThree: THREE.Mesh;
  private materials: THREE.MeshMatcapMaterial[] = [];

  private running = true;
  private cleanups: (() => void)[] = [];

  private interactionMode: InteractionMode = "move";

  // Move state
  private selectedId: number | null = null;
  private dragging = false;
  private movingPlane = new THREE.Plane();
  private lastPoint = new THREE.Vector3();

  // Rotate state
  private isRotating = false;
  private lastMouseX = 0;
  private lastMouseY = 0;

  private raycaster = new THREE.Raycaster();
  private ndc = new THREE.Vector2();

  constructor(
    tf: TF,
    sourceBuffer: ArrayBuffer,
    sourceFilename: string,
    targetBuffer: ArrayBuffer,
    container: HTMLElement,
    isDarkMode = true,
  ) {
    this.tf = tf;
    this.container = container;

    // ── Load and prepare TARGET mesh ────────────────────────────────────
    const target = tf.readStl(targetBuffer);
    centerAndScale(tf, target);
    // Taubin smooth BEFORE creating point cloud (lambda=0.9, kpb=0.1)
    this.smoothedTarget = tf.taubinSmoothed(target, 50, 0.9, 0.1);
    target.delete();

    // ── Create target PointCloud (from smoothed mesh) ──────────────────
    const PointCloud = (tf as any).PointCloud;
    this.targetPC = PointCloud.fromMesh(this.smoothedTarget);
    this.targetPC.buildTree();

    // ── Load and prepare SOURCE mesh ───────────────────────────────────
    const ext = sourceFilename.split(".").pop()?.toLowerCase();
    this.source = ext === "stl" ? tf.readStl(sourceBuffer) : tf.readObj(sourceBuffer);
    centerAndScale(tf, this.source);

    // ── Create source PointCloud ───────────────────────────────────────
    this.sourcePC = PointCloud.fromMesh(this.source);

    // ── Compute AABB diagonal (from source, after centerAndScale) ──────
    {
      const pts = this.source.points;
      const pMin = tf.min(pts, 0);
      const pMax = tf.max(pts, 0);
      const diff = pMax.sub(pMin);
      this.aabbDiagonal = tf.norm(diff) as number;
      pMin.delete(); pMax.delete(); diff.delete(); pts.delete();
    }

    // ── Set identity transformation on target ──────────────────────────
    const identityMat = tf.ndarray(new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ])).reshape([4, 4]);
    this.smoothedTarget.transformation = identityMat;
    this.targetPC.transformation = identityMat;
    identityMat.delete();

    // ── Renderer setup ─────────────────────────────────────────────────
    const bgColor = isDarkMode ? 0x1e1e1e : 0xfafafa;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    container.appendChild(this.renderer.domElement);

    // ── Scene + camera ─────────────────────────────────────────────────
    this.sceneBundle = createScene(this.renderer, { backgroundColor: bgColor, enableFog: false });
    this.switchToOrthographicCamera();

    // ── Target Three.js mesh (teal, semi-transparent) ──────────────────
    const targetGeometry = new THREE.BufferGeometry();
    {
      const pts = this.smoothedTarget.points;
      const fcs = this.smoothedTarget.faces;
      targetGeometry.setAttribute("position", new THREE.BufferAttribute(pts.data, 3));
      targetGeometry.setIndex(new THREE.BufferAttribute(
        new Uint32Array(fcs.data.buffer, fcs.data.byteOffset, fcs.data.length), 1));
      targetGeometry.computeBoundingSphere();
      pts.delete(); fcs.delete();
    }
    const targetColor = new THREE.Color();
    targetColor.setRGB(0, 0.835, 0.745, THREE.SRGBColorSpace);
    const targetMaterial = new THREE.MeshMatcapMaterial({
      side: THREE.DoubleSide,
      flatShading: true,
      color: targetColor,
      transparent: true,
      opacity: 0.5,
    });
    this.materials.push(targetMaterial);
    this.targetThree = new THREE.Mesh(targetGeometry, targetMaterial);
    this.targetThree.matrixAutoUpdate = false;
    this.syncThreeMatrix(this.targetThree, this.smoothedTarget);
    this.sceneBundle.scene.add(this.targetThree);

    // ── Source Three.js mesh (white, opaque) ───────────────────────────
    const sourceGeometry = new THREE.BufferGeometry();
    {
      const pts = this.source.points;
      const fcs = this.source.faces;
      sourceGeometry.setAttribute("position", new THREE.BufferAttribute(pts.data, 3));
      sourceGeometry.setIndex(new THREE.BufferAttribute(
        new Uint32Array(fcs.data.buffer, fcs.data.byteOffset, fcs.data.length), 1));
      sourceGeometry.computeBoundingSphere();
      pts.delete(); fcs.delete();
    }
    const sourceMaterial = new THREE.MeshMatcapMaterial({
      side: THREE.DoubleSide,
      flatShading: true,
    });
    this.materials.push(sourceMaterial);
    this.sourceThree = new THREE.Mesh(sourceGeometry, sourceMaterial);
    this.sourceThree.matrixAutoUpdate = false;
    this.sceneBundle.scene.add(this.sourceThree);

    // ── Matcap textures ────────────────────────────────────────────────
    const matcapUrl = isDarkMode
      ? "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/635D52_A9BCC0_B1AEA0_819598.png"
      : "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/2D2D2F_C6C2C5_727176_94949B.png";
    new THREE.TextureLoader().load(matcapUrl, (tex) => {
      for (let i = 0; i < this.materials.length; i++) {
        this.materials[i]!.matcap = i === 0 ? tex : tex.clone();
        this.materials[i]!.needsUpdate = true;
      }
    });

    // ── Position meshes + fit camera ───────────────────────────────────
    this.positionMeshesForScreen();
    this.fitOrthographicCamera();

    // ── Warmup alignment (run once, then reposition) ───────────────────
    this.align();
    this.positionMeshesForScreen();
    this.fitOrthographicCamera();

    // ── Pointer events ─────────────────────────────────────────────────
    const onPointerMove = (e: PointerEvent) => {
      if (!container.contains(e.target as Node)) return;
      this.updateNDC(e);
      if (this.interactionMode === "rotate" && this.isRotating) {
        this.handleRotation(e);
      } else if (this.dragging && this.selectedId !== null) {
        this.handleDrag();
      } else if (!this.dragging && !this.isRotating) {
        this.handleHover();
      }
    };
    const onPointerDown = (e: PointerEvent) => {
      if (!container.contains(e.target as Node)) return;
      this.updateNDC(e);
      this.handleHover();

      if (this.selectedId !== null) {
        if (this.interactionMode === "rotate") {
          this.isRotating = true;
          this.lastMouseX = e.clientX;
          this.lastMouseY = e.clientY;
        } else {
          this.dragging = true;
        }
        this.sceneBundle.controls.enabled = false;
      }
    };
    const onPointerUp = () => {
      this.dragging = false;
      this.isRotating = false;
      this.selectedId = null;
      this.sceneBundle.controls.enabled = true;
    };

    container.addEventListener("pointermove", onPointerMove);
    container.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointerup", onPointerUp);
    this.cleanups.push(() => {
      container.removeEventListener("pointermove", onPointerMove);
      container.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointerup", onPointerUp);
    });

    // ── Resize observer ────────────────────────────────────────────────
    const resizeObs = new ResizeObserver(() => this.resize());
    resizeObs.observe(container);
    this.cleanups.push(() => resizeObs.disconnect());
    this.resize();

    // ── Animation loop ─────────────────────────────────────────────────
    this.animate();
  }

  // ════════════════════════════════════════════════════════════════════════
  // Orthographic camera
  // ════════════════════════════════════════════════════════════════════════

  private switchToOrthographicCamera() {
    const rect = this.container.getBoundingClientRect();
    const aspect = rect.width / rect.height;
    const frustumSize = 50;

    const orthoCamera = new THREE.OrthographicCamera(
      -frustumSize * aspect / 2, frustumSize * aspect / 2,
      frustumSize / 2, -frustumSize / 2,
      0.1, 1000,
    );
    orthoCamera.position.copy(this.sceneBundle.camera.position);
    orthoCamera.quaternion.copy(this.sceneBundle.camera.quaternion);

    this.sceneBundle.scene.remove(this.sceneBundle.camera);
    this.sceneBundle.scene.add(orthoCamera);

    const oldTarget = ((this.sceneBundle.controls as any).target as THREE.Vector3).clone();
    this.sceneBundle.controls.dispose();
    const newControls = new ArcballControls(
      orthoCamera, this.container.querySelector("canvas")!, this.sceneBundle.scene,
    );
    newControls.rotateSpeed = 1.2;
    newControls.setGizmosVisible(false);
    (newControls as any).target.copy(oldTarget);
    newControls.update();

    (this.sceneBundle as any).camera = orthoCamera;
    (this.sceneBundle as any).controls = newControls;
  }

  private fitOrthographicCamera() {
    const rect = this.container.getBoundingClientRect();
    const aspect = rect.width / rect.height;
    const isLandscape = rect.width > rect.height;
    const camera = this.sceneBundle.camera as unknown as THREE.OrthographicCamera;
    const diag = this.aabbDiagonal;

    // Get positions from both meshes
    const positions: THREE.Vector3[] = [];
    for (const tfMesh of [this.smoothedTarget, this.source]) {
      const m = tfToThree(tfMesh);
      const pos = new THREE.Vector3().setFromMatrixPosition(m);
      positions.push(pos);
    }

    // Compute center and extent
    const center = new THREE.Vector3();
    if (positions.length >= 2) {
      center.addVectors(positions[0]!, positions[1]!).multiplyScalar(0.5);
    }
    const separation = positions.length >= 2 ? positions[0]!.distanceTo(positions[1]!) : 0;
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

    (this.sceneBundle.controls as any).target.copy(center);
    this.sceneBundle.controls.update();
  }

  // ════════════════════════════════════════════════════════════════════════
  // Mesh positioning
  // ════════════════════════════════════════════════════════════════════════

  private positionMeshesForScreen() {
    const rect = this.container.getBoundingClientRect();
    const isLandscape = rect.width > rect.height;
    const diag = this.aabbDiagonal;

    const spacing = isLandscape ? diag * 1.2 : diag * 1.0;
    const offset = isLandscape ? [spacing, 0, 0] : [0, spacing, 0];

    // Target: identity
    const identityMat = this.tf.ndarray(new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ])).reshape([4, 4]);
    this.smoothedTarget.transformation = identityMat;
    this.targetPC.transformation = identityMat;
    identityMat.delete();
    this.syncThreeMatrix(this.targetThree, this.smoothedTarget);

    // Source: offset translation
    const m = new THREE.Matrix4().makeTranslation(offset[0]!, offset[1]!, offset[2]!);
    const mat = threeToTf(this.tf, m);
    this.source.transformation = mat;
    this.sourcePC.transformation = mat;
    mat.delete();
    this.syncThreeMatrix(this.sourceThree, this.source);
  }

  // ════════════════════════════════════════════════════════════════════════
  // Three.js matrix sync
  // ════════════════════════════════════════════════════════════════════════

  private syncThreeMatrix(threeMesh: THREE.Mesh, tfMesh: any) {
    const mat = tfMesh.transformation;
    if (!mat) return;
    const m = new THREE.Matrix4().fromArray(mat.data).transpose();
    threeMesh.matrix.copy(m);
    mat.delete();
  }

  // ════════════════════════════════════════════════════════════════════════
  // Interaction: hover, drag, rotate
  // ════════════════════════════════════════════════════════════════════════

  private updateNDC(e: PointerEvent) {
    const rect = this.container.getBoundingClientRect();
    this.ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  }

  private handleHover() {
    this.raycaster.setFromCamera(this.ndc, this.sceneBundle.camera);
    const o = this.raycaster.ray.origin;
    const d = this.raycaster.ray.direction;
    const tfRay = this.tf.ray([o.x, o.y, o.z, d.x, d.y, d.z]);

    // Only source is pickable
    const hit = pickMesh(this.tf, tfRay, [this.source]);
    tfRay.delete();

    if (hit) {
      this.selectedId = 0; // index in [source] array
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
    this.raycaster.setFromCamera(this.ndc, this.sceneBundle.camera);
    const nextPoint = new THREE.Vector3();
    this.raycaster.ray.intersectPlane(this.movingPlane, nextPoint);
    if (!nextPoint) return;

    const dx = nextPoint.x - this.lastPoint.x;
    const dy = nextPoint.y - this.lastPoint.y;
    const dz = nextPoint.z - this.lastPoint.z;
    this.lastPoint.copy(nextPoint);

    // Update TF transformation: read, modify translation, write back
    const mat = this.source.transformation;
    const dd = mat.data;
    dd[3] += dx; dd[7] += dy; dd[11] += dz;
    this.source.transformation = mat;
    this.sourcePC.transformation = mat;
    mat.delete();

    this.syncThreeMatrix(this.sourceThree, this.source);
  }

  private handleRotation(event: PointerEvent) {
    const dx = event.clientX - this.lastMouseX;
    const dy = event.clientY - this.lastMouseY;
    this.lastMouseX = event.clientX;
    this.lastMouseY = event.clientY;

    const angleX = dy * 0.5; // degrees per pixel
    const angleY = dx * 0.5;

    // Read current transform
    const m = tfToThree(this.source);
    const center = new THREE.Vector3().setFromMatrixPosition(m);

    // Create rotation matrices around world X and Y axes
    const rotX = new THREE.Matrix4().makeRotationAxis(
      new THREE.Vector3(1, 0, 0), THREE.MathUtils.degToRad(angleX),
    );
    const rotY = new THREE.Matrix4().makeRotationAxis(
      new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(angleY),
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

    // Write back
    const newMat = threeToTf(this.tf, newMatrix);
    this.source.transformation = newMat;
    this.sourcePC.transformation = newMat;
    newMat.delete();

    this.syncThreeMatrix(this.sourceThree, this.source);
  }

  // ════════════════════════════════════════════════════════════════════════
  // Alignment: OBB + ICP
  // ════════════════════════════════════════════════════════════════════════

  public align(): number {
    const t0 = performance.now();

    // Read current source transform
    const mCurrent = tfToThree(this.source);

    // Stage 1: OBB coarse alignment
    const T_obb = this.tf.fitObbAlignment(this.sourcePC, this.targetPC);
    const mObb = new THREE.Matrix4().fromArray(T_obb.data).transpose();
    T_obb.delete();

    // Compose: T_after_obb = delta_obb * T_current
    const mAfterObb = mObb.multiply(mCurrent);

    // Apply OBB result so ICP sees aligned position
    let mat = threeToTf(this.tf, mAfterObb);
    this.sourcePC.transformation = mat;
    mat.delete();

    // Stage 2: ICP refinement (point-to-plane)
    const T_icp = this.tf.fitIcpAlignment(this.sourcePC, this.targetPC, {
      maxIterations: 50,
      nSamples: 1000,
      k: 1,
    });
    const mIcp = new THREE.Matrix4().fromArray(T_icp.data).transpose();
    T_icp.delete();

    // Compose: T_final = delta_icp * T_after_obb
    const mFinal = mIcp.multiply(mAfterObb);

    // Apply final transform to both source mesh and PC
    mat = threeToTf(this.tf, mFinal);
    this.source.transformation = mat;
    this.sourcePC.transformation = mat;
    mat.delete();

    this.syncThreeMatrix(this.sourceThree, this.source);
    return performance.now() - t0;
  }

  // ════════════════════════════════════════════════════════════════════════
  // Public API
  // ════════════════════════════════════════════════════════════════════════

  public setMode(mode: InteractionMode) {
    this.interactionMode = mode;
  }

  public getMode(): InteractionMode {
    return this.interactionMode;
  }

  public applyTheme(isDark: boolean) {
    this.sceneBundle.scene.background = new THREE.Color(isDark ? 0x1e1e1e : 0xfafafa);

    const matcapUrl = isDark
      ? "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/635D52_A9BCC0_B1AEA0_819598.png"
      : "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/2D2D2F_C6C2C5_727176_94949B.png";
    new THREE.TextureLoader().load(matcapUrl, (tex) => {
      for (let i = 0; i < this.materials.length; i++) {
        const mat = this.materials[i]!;
        if (mat.matcap) mat.matcap.dispose();
        mat.matcap = i === 0 ? tex : tex.clone();
        mat.needsUpdate = true;
      }
    });
  }

  // ════════════════════════════════════════════════════════════════════════
  // Resize + animate
  // ════════════════════════════════════════════════════════════════════════

  private resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    const cam = this.sceneBundle.camera as unknown as THREE.OrthographicCamera;
    const frustumH = cam.top - cam.bottom;
    const aspect = w / h;
    cam.left = -frustumH * aspect / 2;
    cam.right = frustumH * aspect / 2;
    cam.updateProjectionMatrix();
  }

  private animate() {
    if (!this.running) return;
    requestAnimationFrame(() => this.animate());
    this.sceneBundle.controls.update();
    this.renderer.render(this.sceneBundle.scene, this.sceneBundle.camera);
  }

  // ════════════════════════════════════════════════════════════════════════
  // Dispose
  // ════════════════════════════════════════════════════════════════════════

  public dispose() {
    this.running = false;
    for (const fn of this.cleanups) fn();
    this.cleanups = [];

    this.sceneBundle.controls.dispose();
    this.targetThree.geometry.dispose();
    this.sourceThree.geometry.dispose();
    for (const mat of this.materials) mat.dispose();

    this.renderer.dispose();
    this.renderer.domElement.remove();

    this.sourcePC.delete();
    this.targetPC.delete();
    this.source.delete();
    this.smoothedTarget.delete();
  }
}

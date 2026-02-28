import { syncOrbitControls, createScene, type SceneBundle } from "@/utils/sceneUtils";
import { centerAndScale, pickMesh, randomTransformation, RollingAverage, CurveRenderer } from "@/utils/utils";
import * as THREE from "three";
import { ArcballControls } from "three/addons/controls/ArcballControls.js";

type TF = typeof import("@/examples/trueform/index.js");

export class BooleanExample {
  private tf: TF;
  private dragon: any;
  private sphere: any;
  private tfMeshes: any[] = []; // [0] = dragon, [1] = sphere
  private resultMesh: any = null;

  private container1: HTMLElement;
  private container2: HTMLElement;
  private renderer1: THREE.WebGLRenderer;
  private renderer2: THREE.WebGLRenderer;
  private sceneBundle1: SceneBundle;
  private sceneBundle2: SceneBundle;

  private dragonThree: THREE.Mesh;
  private sphereThree: THREE.Mesh;
  private resultThree: THREE.Mesh;
  private resultGeometry: THREE.BufferGeometry;
  private curveRenderer: CurveRenderer;

  private materials: THREE.MeshMatcapMaterial[] = [];
  private running = true;
  private cleanups: (() => void)[] = [];

  private selectedId: number | null = null;
  private dragging = false;
  private movingPlane = new THREE.Plane();
  private lastPoint = new THREE.Vector3();
  private keyPressed = false;
  private lastActiveScene: 1 | 2 = 1;
  private isSyncing = false;

  private sphereScale = 2.0;
  private booleanTiming = new RollingAverage();
  private raycaster = new THREE.Raycaster();
  private ndc = new THREE.Vector2();

  public refreshTimeValue: (() => number) | null = null;
  public onSphereSizeDelta?: (deltaSteps: number) => void;

  constructor(
    tf: TF,
    fileBuffer: ArrayBuffer,
    fileName: string,
    container: HTMLElement,
    container2: HTMLElement,
    isDarkMode = true,
  ) {
    this.tf = tf;
    this.container1 = container;
    this.container2 = container2;

    // ── Load dragon ──────────────────────────────────────────────────────
    const ext = fileName.split(".").pop()?.toLowerCase();
    this.dragon = ext === "stl" ? tf.readStl(fileBuffer) : tf.readObj(fileBuffer);
    centerAndScale(tf, this.dragon);

    // Set identity transformation (required for booleanDifference to tag it)
    const dragonMat = tf.ndarray(new Float32Array([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    ])).reshape([4, 4]);
    this.dragon.transformation = dragonMat;
    dragonMat.delete();

    // ── Sphere position (C++ uses obb.axes[1] * 4.0; OBB not exposed in TS) ─
    // After centerAndScale the dragon is deterministic: OBB 2nd axis ≈ Y.
    const spherePos = [2.5, 1.5, -1.2];

    // ── Create sphere ────────────────────────────────────────────────────
    this.sphere = tf.sphereMesh(1.0, 32, 32);

    const s = this.sphereScale;
    const sphereMat = tf.ndarray(new Float32Array([
      s, 0, 0, spherePos[0]!, 0, s, 0, spherePos[1]!, 0, 0, s, spherePos[2]!, 0, 0, 0, 1,
    ])).reshape([4, 4]);
    this.sphere.transformation = sphereMat;
    sphereMat.delete();

    this.tfMeshes = [this.dragon, this.sphere];

    // ── Two renderers + scenes ───────────────────────────────────────────
    const bgColor1 = isDarkMode ? 0x1e1e1e : 0xfafafa;
    const bgColor2 = isDarkMode ? 0x262626 : 0xf5f5f5;

    this.renderer1 = new THREE.WebGLRenderer({ antialias: true });
    this.renderer1.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer1.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer1.toneMappingExposure = 1.0;
    container.appendChild(this.renderer1.domElement);
    this.sceneBundle1 = createScene(this.renderer1, { backgroundColor: bgColor1, enableFog: false });

    this.renderer2 = new THREE.WebGLRenderer({ antialias: true });
    this.renderer2.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer2.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer2.toneMappingExposure = 1.0;
    container2.appendChild(this.renderer2.domElement);
    this.sceneBundle2 = createScene(this.renderer2, { backgroundColor: bgColor2, enableFog: false });

    // Switch both to orthographic cameras
    this.switchToOrthographicCamera(this.sceneBundle1, container);
    this.switchToOrthographicCamera(this.sceneBundle2, container2);

    // ── Matcap texture ───────────────────────────────────────────────────
    const matcapUrl = isDarkMode
      ? "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/635D52_A9BCC0_B1AEA0_819598.png"
      : "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/2D2D2F_C6C2C5_727176_94949B.png";

    // ── Dragon Three.js mesh (flat shading, white) ───────────────────────
    const dragonGeometry = new THREE.BufferGeometry();
    {
      const pts = this.dragon.points;
      const fcs = this.dragon.faces;
      dragonGeometry.setAttribute("position", new THREE.BufferAttribute(pts.data, 3));
      dragonGeometry.setIndex(new THREE.BufferAttribute(
        new Uint32Array(fcs.data.buffer, fcs.data.byteOffset, fcs.data.length), 1));
      dragonGeometry.computeBoundingSphere();
      pts.delete(); fcs.delete();
    }
    const dragonMaterial = new THREE.MeshMatcapMaterial({
      side: THREE.DoubleSide,
      flatShading: true,
    });
    this.materials.push(dragonMaterial);
    this.dragonThree = new THREE.Mesh(dragonGeometry, dragonMaterial);
    this.dragonThree.matrixAutoUpdate = false;
    this.syncThreeMatrix(0, this.dragonThree);
    this.sceneBundle1.scene.add(this.dragonThree);

    // ── Sphere Three.js mesh (smooth shading, light blue) ────────────────
    const sphereGeometry = new THREE.BufferGeometry();
    {
      const pts = this.sphere.points;
      const fcs = this.sphere.faces;
      sphereGeometry.setAttribute("position", new THREE.BufferAttribute(pts.data, 3));
      sphereGeometry.setIndex(new THREE.BufferAttribute(
        new Uint32Array(fcs.data.buffer, fcs.data.byteOffset, fcs.data.length), 1));
      sphereGeometry.computeVertexNormals();
      sphereGeometry.computeBoundingSphere();
      pts.delete(); fcs.delete();
    }
    const sphereColor = new THREE.Color();
    sphereColor.setRGB(0.7, 0.85, 1.0, THREE.SRGBColorSpace);
    const sphereMaterial = new THREE.MeshMatcapMaterial({
      side: THREE.DoubleSide,
      flatShading: false,
      color: sphereColor,
    });
    this.materials.push(sphereMaterial);
    this.sphereThree = new THREE.Mesh(sphereGeometry, sphereMaterial);
    this.sphereThree.matrixAutoUpdate = false;
    this.syncThreeMatrix(1, this.sphereThree);
    this.sceneBundle1.scene.add(this.sphereThree);

    // ── CurveRenderer (teal, instanced tubes) ────────────────────────────
    this.curveRenderer = new CurveRenderer({
      color: 0x00d5be,
      radius: 0.075,
      maxSegments: 20000,
    });
    this.sceneBundle1.scene.add(this.curveRenderer.object);

    // ── Result Three.js mesh (flat shading, right scene) ─────────────────
    this.resultGeometry = new THREE.BufferGeometry();
    const resultMaterial = new THREE.MeshMatcapMaterial({
      side: THREE.DoubleSide,
      flatShading: true,
    });
    this.materials.push(resultMaterial);
    this.resultThree = new THREE.Mesh(this.resultGeometry, resultMaterial);
    this.resultThree.matrixAutoUpdate = false;
    this.sceneBundle2.scene.add(this.resultThree);

    // Load matcap textures for all materials
    new THREE.TextureLoader().load(matcapUrl, (tex) => {
      for (let i = 0; i < this.materials.length; i++) {
        this.materials[i]!.matcap = i === 0 ? tex : tex.clone();
        this.materials[i]!.needsUpdate = true;
      }
    });

    // ── Initial boolean computation (warm-up, discard timing) ───────────
    this.computeBoolean();
    this.booleanTiming.clear();

    // ── Fit cameras ──────────────────────────────────────────────────────
    this.fitOrthoCameraToMeshes(this.sceneBundle1, 1.8);

    // Adjust camera angle slightly right and up for better dragon view
    const camera = this.sceneBundle1.camera;
    const target = (this.sceneBundle1.controls as any).target as THREE.Vector3;
    const offset = camera.position.clone().sub(target);
    const dist = offset.length();
    const angleH = 0.15;
    const angleV = 0.1;
    camera.position.set(
      target.x + dist * Math.sin(angleH),
      target.y + dist * Math.sin(angleV),
      target.z + dist * Math.cos(angleH) * Math.cos(angleV),
    );
    camera.lookAt(target);
    this.sceneBundle1.controls.update();

    this.fitOrthoCameraToMeshes(this.sceneBundle2, 1.8);
    syncOrbitControls(this.sceneBundle1.controls, this.sceneBundle2.controls);

    // ── Bidirectional camera sync ─────────────────────────────────────────
    const syncFrom = (source: 1 | 2) => {
      if (this.isSyncing) return;
      this.isSyncing = true;
      if (source === 1) {
        syncOrbitControls(this.sceneBundle1.controls, this.sceneBundle2.controls);
      } else {
        syncOrbitControls(this.sceneBundle2.controls, this.sceneBundle1.controls);
      }
      this.isSyncing = false;
    };

    this.sceneBundle1.controls.addEventListener("change", () => {
      if (this.lastActiveScene === 1) syncFrom(1);
    });
    this.sceneBundle2.controls.addEventListener("change", () => {
      if (this.lastActiveScene === 2) syncFrom(2);
    });

    const onEnterContainer1 = () => { this.lastActiveScene = 1; };
    const onEnterContainer2 = () => { this.lastActiveScene = 2; };
    container.addEventListener("pointerenter", onEnterContainer1);
    container2.addEventListener("pointerenter", onEnterContainer2);
    this.cleanups.push(() => {
      container.removeEventListener("pointerenter", onEnterContainer1);
      container2.removeEventListener("pointerenter", onEnterContainer2);
    });

    // ── Pointer events (on left container) ───────────────────────────────
    const onPointerMove = (e: PointerEvent) => {
      if (!container.contains(e.target as Node)) return;
      this.updateNDC(e);
      if (this.dragging && this.selectedId !== null) {
        this.handleDrag();
      } else if (!this.dragging) {
        this.handleHover();
      }
    };
    const onPointerDown = (e: PointerEvent) => {
      if (!container.contains(e.target as Node)) return;
      this.updateNDC(e);
      this.handleHover();
      if (this.selectedId !== null) {
        this.dragging = true;
        this.sceneBundle1.controls.enabled = false;
      }
    };
    const onPointerUp = () => {
      this.dragging = false;
      this.selectedId = null;
      this.sceneBundle1.controls.enabled = true;
    };

    container.addEventListener("pointermove", onPointerMove);
    container.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointerup", onPointerUp);
    this.cleanups.push(() => {
      container.removeEventListener("pointermove", onPointerMove);
      container.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointerup", onPointerUp);
    });

    // ── Keyboard events ──────────────────────────────────────────────────
    const onKeyDown = (event: KeyboardEvent) => {
      if (this.keyPressed) return;
      this.keyPressed = true;
      if (event.key === "r") { this.resyncCamera(); return; }
      if (event.key === "n") { this.randomize(); return; }
    };
    const onKeyUp = () => { this.keyPressed = false; };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    this.cleanups.push(() => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    });

    // ── Ctrl+scroll for sphere size ──────────────────────────────────────
    const onWheel = (event: WheelEvent) => {
      if (!event.ctrlKey) return;
      event.preventDefault();
      const delta = event.deltaY !== 0 ? event.deltaY : event.deltaX;
      if (delta === 0) return;
      const normalizedDelta = delta / Math.abs(delta);
      const handled = this.adjustSphereSize(-normalizedDelta);
      if (handled) {
        this.onSphereSizeDelta?.(-normalizedDelta);
        event.stopImmediatePropagation();
      }
    };
    const wheelOpts = { passive: false, capture: true } as const;
    window.addEventListener("wheel", onWheel, wheelOpts);
    this.cleanups.push(() => {
      window.removeEventListener("wheel", onWheel, wheelOpts);
    });

    // ── Resize observers ─────────────────────────────────────────────────
    const resizeObs1 = new ResizeObserver(() => this.resize1());
    resizeObs1.observe(container);
    const resizeObs2 = new ResizeObserver(() => this.resize2());
    resizeObs2.observe(container2);
    this.cleanups.push(() => { resizeObs1.disconnect(); resizeObs2.disconnect(); });
    this.resize1();
    this.resize2();

    // ── Animation loop ───────────────────────────────────────────────────
    this.animate();
  }

  // ════════════════════════════════════════════════════════════════════════
  // Orthographic camera helpers
  // ════════════════════════════════════════════════════════════════════════

  private switchToOrthographicCamera(sceneBundle: SceneBundle, container: HTMLElement) {
    const rect = container.getBoundingClientRect();
    const aspect = rect.width / rect.height;
    const frustumSize = 50;

    const orthoCamera = new THREE.OrthographicCamera(
      -frustumSize * aspect / 2, frustumSize * aspect / 2,
      frustumSize / 2, -frustumSize / 2,
      0.1, 1000,
    );
    orthoCamera.position.copy(sceneBundle.camera.position);
    orthoCamera.quaternion.copy(sceneBundle.camera.quaternion);

    sceneBundle.scene.remove(sceneBundle.camera);
    sceneBundle.scene.add(orthoCamera);

    const oldTarget = ((sceneBundle.controls as any).target as THREE.Vector3).clone();
    sceneBundle.controls.dispose();
    const newControls = new ArcballControls(orthoCamera, container.querySelector("canvas")!, sceneBundle.scene);
    newControls.rotateSpeed = 1.2;
    newControls.setGizmosVisible(false);
    (newControls as any).target.copy(oldTarget);
    newControls.update();

    (sceneBundle as any).camera = orthoCamera;
    (sceneBundle as any).controls = newControls;
  }

  private fitOrthoCameraToMeshes(sceneBundle: SceneBundle, offset = 1.25) {
    const { scene, controls } = sceneBundle;
    const camera = sceneBundle.camera as unknown as THREE.OrthographicCamera;

    const meshes: THREE.Mesh[] = [];
    scene.traverse((child) => {
      if (child.type === "Mesh" || child.type === "InstancedMesh") {
        meshes.push(child as THREE.Mesh);
      }
    });
    if (meshes.length === 0) return;

    const combinedBox = new THREE.Box3();
    meshes.forEach((mesh) => { combinedBox.union(new THREE.Box3().setFromObject(mesh)); });

    const center = combinedBox.getCenter(new THREE.Vector3());
    const size = combinedBox.getSize(new THREE.Vector3());
    const maxDimension = Math.max(size.x, size.y, size.z) * offset;

    const aspect = (camera.right - camera.left) / (camera.top - camera.bottom);
    camera.left = -maxDimension * aspect / 2;
    camera.right = maxDimension * aspect / 2;
    camera.top = maxDimension / 2;
    camera.bottom = -maxDimension / 2;

    const distance = maxDimension * 2;
    camera.position.set(center.x, center.y, center.z + distance);
    camera.lookAt(center);
    camera.updateProjectionMatrix();

    (controls as any).target.copy(center);
    controls.update();
  }

  // ════════════════════════════════════════════════════════════════════════
  // Boolean computation
  // ════════════════════════════════════════════════════════════════════════

  private computeBoolean() {

    if (this.resultMesh) { this.resultMesh.delete(); this.resultMesh = null; }

    const t0 = performance.now();
    const result = this.tf.booleanDifference(this.dragon, this.sphere, { returnCurves: true });
    this.booleanTiming.add(performance.now() - t0);

    // Update result geometry — views, zero copies
    const rPts = result.mesh.points;
    const rFaces = result.mesh.faces;
    this.resultGeometry.setAttribute("position", new THREE.BufferAttribute(rPts.data, 3));
    this.resultGeometry.setIndex(new THREE.BufferAttribute(
      new Uint32Array(rFaces.data.buffer, rFaces.data.byteOffset, rFaces.data.length), 1));
    this.resultGeometry.computeBoundingSphere();
    this.resultGeometry.computeBoundingBox();
    rPts.delete(); rFaces.delete();

    // Update curves — zero-copy via updateFromBuffers
    const pts = result.curves.points;
    const pathsBuf = result.curves.paths;
    const pathsData = pathsBuf.data;
    const pathsOffsets = pathsBuf.offsets;
    this.curveRenderer.updateFromBuffers(
      pts.data as Float32Array,
      pathsData.data as Int32Array,
      pathsOffsets.data as Int32Array,
    );
    pathsOffsets.delete(); pathsData.delete(); pathsBuf.delete(); pts.delete();

    this.resultMesh = result.mesh;
    result.labels.delete();
    result.curves.delete();

    if (this.refreshTimeValue) this.refreshTimeValue();
  }

  // ════════════════════════════════════════════════════════════════════════
  // Three.js matrix sync
  // ════════════════════════════════════════════════════════════════════════

  private syncThreeMatrix(index: number, mesh?: THREE.Mesh) {
    const target = mesh ?? (index === 0 ? this.dragonThree : this.sphereThree);
    const mat = this.tfMeshes[index]!.transformation;
    if (!mat) return;
    const m = new THREE.Matrix4();
    m.fromArray(mat.data);
    m.transpose();
    target.matrix.copy(m);
    mat.delete();
  }

  // ════════════════════════════════════════════════════════════════════════
  // Interaction: drag
  // ════════════════════════════════════════════════════════════════════════

  private updateNDC(e: PointerEvent) {
    const rect = this.container1.getBoundingClientRect();
    this.ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  }

  private handleHover() {
    this.raycaster.setFromCamera(this.ndc, this.sceneBundle1.camera);
    const o = this.raycaster.ray.origin;
    const d = this.raycaster.ray.direction;
    const tfRay = this.tf.ray([o.x, o.y, o.z, d.x, d.y, d.z]);

    const hit = pickMesh(this.tf, tfRay, this.tfMeshes);
    tfRay.delete();

    if (hit) {
      this.selectedId = hit.index;
      const hitPoint = this.raycaster.ray.at(hit.t, new THREE.Vector3());
      const camDir = new THREE.Vector3();
      this.sceneBundle1.camera.getWorldDirection(camDir);
      this.movingPlane.setFromNormalAndCoplanarPoint(camDir, hitPoint);
      this.lastPoint.copy(hitPoint);
    } else {
      this.selectedId = null;
    }
  }

  private handleDrag() {
    const id = this.selectedId!;

    this.raycaster.setFromCamera(this.ndc, this.sceneBundle1.camera);
    const nextPoint = new THREE.Vector3();
    this.raycaster.ray.intersectPlane(this.movingPlane, nextPoint);
    if (!nextPoint) return;

    const dx = nextPoint.x - this.lastPoint.x;
    const dy = nextPoint.y - this.lastPoint.y;
    const dz = nextPoint.z - this.lastPoint.z;
    this.lastPoint.copy(nextPoint);

    // Update TF transformation: read, modify translation, write back
    const mat = this.tfMeshes[id]!.transformation;
    const d = mat.data;
    d[3] += dx; d[7] += dy; d[11] += dz;
    this.tfMeshes[id]!.transformation = mat;
    mat.delete();

    this.syncThreeMatrix(id);
    this.computeBoolean();
  }

  // ════════════════════════════════════════════════════════════════════════
  // Public API
  // ════════════════════════════════════════════════════════════════════════

  public getAverageTime(): number {
    return this.booleanTiming.average;
  }

  public resyncCamera() {
    syncOrbitControls(this.sceneBundle1.controls, this.sceneBundle2.controls);
  }

  public randomize() {
    const tf = this.tf;

    // Dragon: random rotation, preserve translation
    const dMat = this.dragon.transformation;
    const dtx = dMat.data[3], dty = dMat.data[7], dtz = dMat.data[11];
    dMat.delete();
    const dNew = randomTransformation(tf, dtx, dty, dtz);
    this.dragon.transformation = dNew;
    dNew.delete();

    // Sphere: random rotation with scale, preserve translation
    const sMat = this.sphere.transformation;
    const stx = sMat.data[3], sty = sMat.data[7], stz = sMat.data[11];
    sMat.delete();
    const sNew = randomTransformation(tf, stx, sty, stz);
    const sd = sNew.data;
    const sc = this.sphereScale;
    sd[0] *= sc; sd[1] *= sc; sd[2] *= sc;
    sd[4] *= sc; sd[5] *= sc; sd[6] *= sc;
    sd[8] *= sc; sd[9] *= sc; sd[10] *= sc;
    this.sphere.transformation = sNew;
    sNew.delete();

    this.syncThreeMatrix(0);
    this.syncThreeMatrix(1);
    this.computeBoolean();
  }

  public adjustSphereSize(deltaSteps: number): boolean {
    const steps = Math.trunc(deltaSteps);
    if (steps === 0) return false;
    const newScale = Math.max(0.1, Math.min(5.0, this.sphereScale + steps * 0.05));
    if (newScale === this.sphereScale) return false;
    this.sphereScale = newScale;

    // Set only diagonal — matches old WASM behavior (matrix[0/5/10] = scale)
    const mat = this.sphere.transformation;
    const d = mat.data;
    d[0] = newScale; d[5] = newScale; d[10] = newScale;
    this.sphere.transformation = mat;
    mat.delete();

    this.syncThreeMatrix(1);
    this.computeBoolean();
    return true;
  }

  public applyTheme(isDark: boolean) {
    this.sceneBundle1.scene.background = new THREE.Color(isDark ? 0x1e1e1e : 0xfafafa);
    this.sceneBundle2.scene.background = new THREE.Color(isDark ? 0x262626 : 0xf5f5f5);

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

  private resize1() {
    const w = this.container1.clientWidth;
    const h = this.container1.clientHeight;
    this.renderer1.setSize(w, h);
    this.renderer1.setPixelRatio(window.devicePixelRatio);
    const cam = this.sceneBundle1.camera as unknown as THREE.OrthographicCamera;
    const frustumH = cam.top - cam.bottom;
    const aspect = w / h;
    cam.left = -frustumH * aspect / 2;
    cam.right = frustumH * aspect / 2;
    cam.updateProjectionMatrix();
  }

  private resize2() {
    const w = this.container2.clientWidth;
    const h = this.container2.clientHeight;
    this.renderer2.setSize(w, h);
    this.renderer2.setPixelRatio(window.devicePixelRatio);
    const cam = this.sceneBundle2.camera as unknown as THREE.OrthographicCamera;
    const frustumH = cam.top - cam.bottom;
    const aspect = w / h;
    cam.left = -frustumH * aspect / 2;
    cam.right = frustumH * aspect / 2;
    cam.updateProjectionMatrix();
  }

  private animate() {
    if (!this.running) return;
    requestAnimationFrame(() => this.animate());
    this.sceneBundle1.controls.update();
    this.sceneBundle2.controls.update();
    this.renderer1.render(this.sceneBundle1.scene, this.sceneBundle1.camera);
    this.renderer2.render(this.sceneBundle2.scene, this.sceneBundle2.camera);
  }

  public dispose() {
    this.running = false;
    for (const fn of this.cleanups) fn();
    this.cleanups = [];

    this.sceneBundle1.controls.dispose();
    this.sceneBundle2.controls.dispose();
    this.curveRenderer.dispose();

    this.dragonThree.geometry.dispose();
    this.sphereThree.geometry.dispose();
    this.resultGeometry.dispose();
    for (const mat of this.materials) mat.dispose();

    this.renderer1.dispose();
    this.renderer1.domElement.remove();
    this.renderer2.dispose();
    this.renderer2.domElement.remove();

    if (this.resultMesh) { this.resultMesh.delete(); this.resultMesh = null; }
    this.sphere.delete();
    this.dragon.delete();
  }
}

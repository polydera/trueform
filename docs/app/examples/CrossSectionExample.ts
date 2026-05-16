import { fitCameraToAllMeshesFromZPlane, createScene, type SceneBundle } from "@/utils/sceneUtils";
import { centerAndScale, CurveRenderer, RollingAverage } from "@/utils/utils";
import * as THREE from "three";

type TF = typeof import("@polydera/trueform");

export class CrossSectionExample {
  private tf: TF;
  private mesh: any;
  private container: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private sceneBundle: SceneBundle;
  private curveRenderer: CurveRenderer;
  private baseMesh: THREE.Mesh;
  private crossSectionMesh: THREE.Mesh;
  private running = true;
  private cleanups: (() => void)[] = [];

  // Scalar field (NDArray [N], persists across scroll events)
  private scalarsND: any = null;
  private scalarMin = 0;
  private scalarMax = 1;
  private planeOffset = 0;
  private normalVec: any = null;
  private timing = new RollingAverage();

  public refreshTimeValue: (() => number) | null = null;

  constructor(tf: TF, fileBuffer: ArrayBuffer, fileName: string, container: HTMLElement, isDarkMode = true) {
    this.tf = tf;
    this.container = container;

    // Load mesh
    const ext = fileName.split(".").pop()?.toLowerCase();
    this.mesh = ext === "stl" ? tf.readStl(fileBuffer) : tf.readObj(fileBuffer);

    // Center and scale mesh (AABB center at origin, diagonal/2 = 10)
    centerAndScale(this.tf, this.mesh);

    // Initial normal [1, 2, 1] normalized
    this.normalVec = tf.normalize(tf.vector(1, 2, 1));

    // Compute initial scalar field
    this.computeScalars();

    // Three.js setup
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    container.appendChild(this.renderer.domElement);

    this.sceneBundle = createScene(this.renderer, {
      backgroundColor: getBrandBackground(),
      enableFog: false,
    });

    // Base mesh (semi-transparent)
    const points = this.mesh.points;
    const faces = this.mesh.faces;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(points.data, 3));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(faces.data.buffer, faces.data.byteOffset, faces.data.length), 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();

    const matcapUrl = isDarkMode
      ? "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/635D52_A9BCC0_B1AEA0_819598.png"
      : "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/2D2D2F_C6C2C5_727176_94949B.png";
    const baseMaterial = new THREE.MeshMatcapMaterial({
      side: THREE.DoubleSide,
      flatShading: true,
      transparent: true,
      opacity: 0.25,
      depthWrite: false,
    });
    this.baseMesh = new THREE.Mesh(geometry, baseMaterial);
    this.baseMesh.matrixAutoUpdate = false;
    this.sceneBundle.scene.add(this.baseMesh);

    // Cross-section cap mesh
    const capMaterial = new THREE.MeshMatcapMaterial({
      side: THREE.DoubleSide,
      flatShading: true,
      color: new THREE.Color(0x00a89a),
    });
    this.crossSectionMesh = new THREE.Mesh(new THREE.BufferGeometry(), capMaterial);
    this.crossSectionMesh.matrixAutoUpdate = false;
    this.sceneBundle.scene.add(this.crossSectionMesh);

    // Load matcap texture
    new THREE.TextureLoader().load(matcapUrl, (tex) => {
      baseMaterial.matcap = tex;
      baseMaterial.needsUpdate = true;
      capMaterial.matcap = tex.clone();
      capMaterial.needsUpdate = true;
    });

    // Curve renderer
    this.curveRenderer = new CurveRenderer({ color: 0x00d5be, radius: 0.075, maxSegments: 20000 });
    this.sceneBundle.scene.add(this.curveRenderer.object);

    // Compute initial cross-section
    this.recomputeCrossSection();

    // Fit camera
    fitCameraToAllMeshesFromZPlane(this.sceneBundle, 1.5);
    this.sceneBundle.controls.enableZoom = false;

    // Scroll interaction (matches WASM cross_section_web.h:104)
    const range = () => this.scalarMax - this.scalarMin;
    const margin = () => range() * 0.01;
    const scrollStep = (delta: number) => {
      this.planeOffset += delta * 0.003 * range();
      this.planeOffset = Math.max(
        this.scalarMin + margin(),
        Math.min(this.scalarMax - margin(), this.planeOffset),
      );
      this.recomputeCrossSection();
    };

    const onWheel = (event: WheelEvent) => {
      if (!container.contains(event.target as Node)) return;
      event.preventDefault();
      const delta = event.deltaY !== 0 ? event.deltaY : event.deltaX;
      if (delta === 0) return;
      scrollStep(Math.sign(delta));
    };
    window.addEventListener("wheel", onWheel, { passive: false, capture: true });
    this.cleanups.push(() => window.removeEventListener("wheel", onWheel, { passive: false, capture: true } as any));

    // Touch scroll
    let touchActive = false;
    let lastTouchY = 0;
    const onTouchStart = (e: TouchEvent) => {
      if (e.touches.length !== 1 || !container.contains(e.target as Node)) return;
      e.preventDefault();
      touchActive = true;
      this.sceneBundle.controls.enabled = false;
      lastTouchY = e.touches[0]!.clientY;
    };
    const onTouchMove = (e: TouchEvent) => {
      if (!touchActive || e.touches.length !== 1) { touchActive = false; this.sceneBundle.controls.enabled = true; return; }
      e.preventDefault();
      const dy = e.touches[0]!.clientY - lastTouchY;
      if (Math.abs(dy) < 10) return;
      scrollStep(Math.sign(dy));
      lastTouchY = e.touches[0]!.clientY;
    };
    const onTouchEnd = () => { touchActive = false; this.sceneBundle.controls.enabled = true; };
    const touchOpts = { passive: false, capture: true };
    window.addEventListener("touchstart", onTouchStart, touchOpts);
    window.addEventListener("touchmove", onTouchMove, touchOpts);
    window.addEventListener("touchend", onTouchEnd, touchOpts);
    window.addEventListener("touchcancel", onTouchEnd, touchOpts);
    this.cleanups.push(() => {
      window.removeEventListener("touchstart", onTouchStart, touchOpts as any);
      window.removeEventListener("touchmove", onTouchMove, touchOpts as any);
      window.removeEventListener("touchend", onTouchEnd, touchOpts as any);
      window.removeEventListener("touchcancel", onTouchEnd, touchOpts as any);
    });

    // Resize
    const resizeObs = new ResizeObserver(() => this.resize());
    resizeObs.observe(container);
    this.cleanups.push(() => resizeObs.disconnect());
    this.resize();

    // Animate
    this.animate();
  }

  private computeScalars() {
    const tf = this.tf;
    if (this.scalarsND) { this.scalarsND.delete(); this.scalarsND = null; }

    const points = this.mesh.points;
    const centroid = tf.mean(points, 0) as any;
    const d = -(tf.dot(this.normalVec, centroid) as number);
    const p = tf.plane(this.normalVec, d);
    const pts = tf.point(points);

    this.scalarsND = tf.distance(pts, p);
    this.scalarMin = this.scalarsND.min() as number;
    this.scalarMax = this.scalarsND.max() as number;
    this.planeOffset = (this.scalarMin + this.scalarMax) * 0.5;

    centroid.delete();
  }

  private recomputeCrossSection() {
    const t0 = performance.now();
    const tf = this.tf;

    // 1. Isocontours at the cut value → boundary curves
    const curves = tf.isocontours(this.mesh, this.scalarsND, this.planeOffset);

    // 2. Triangulate curve paths as polygons → filled cap mesh
    const cap = tf.triangulate({ faces: curves.paths, points: curves.points });

    this.timing.add(performance.now() - t0);

    // 3. Update cap mesh geometry
    const capPoints = cap.points;
    const capFaces = cap.faces;
    const geom = this.crossSectionMesh.geometry;
    geom.setAttribute("position", new THREE.BufferAttribute(capPoints.data, 3));
    geom.setIndex(new THREE.BufferAttribute(
      new Uint32Array(capFaces.data.buffer, capFaces.data.byteOffset, capFaces.data.length), 1,
    ));
    if (capPoints.data.length >= 3) {
      geom.computeBoundingSphere();
      geom.computeBoundingBox();
    }

    // 4. Update curves for tube rendering
    const curvePts = curves.points.data as Float32Array;
    const paths: number[][] = [];
    for (const p of curves.paths) {
      paths.push(Array.from(p.data));
      p.delete();
    }
    this.curveRenderer.update({ points: curvePts, paths });

    // Cleanup
    cap.delete();
    curves.delete();
    if (this.refreshTimeValue) this.refreshTimeValue();
  }

  public randomize() {
    const tf = this.tf;
    this.normalVec.delete();
    this.normalVec = tf.normalize(tf.vector(tf.random("float32", [3], -1, 1)));
    this.computeScalars();
    this.recomputeCrossSection();
  }

  public getAverageTime(): number {
    return this.timing.average;
  }

  public applyTheme(isDark: boolean) {
    this.sceneBundle.scene.background = null;

    const matcapUrl = isDark
      ? "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/635D52_A9BCC0_B1AEA0_819598.png"
      : "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/2D2D2F_C6C2C5_727176_94949B.png";
    const newTex = new THREE.TextureLoader().load(matcapUrl);

    const baseMat = this.baseMesh.material as THREE.MeshMatcapMaterial;
    if (baseMat.matcap) baseMat.matcap.dispose();
    baseMat.matcap = newTex;

    const capMat = this.crossSectionMesh.material as THREE.MeshMatcapMaterial;
    if (capMat.matcap) capMat.matcap.dispose();
    capMat.matcap = newTex.clone();
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
    this.sceneBundle.controls.update();
    this.renderer.render(this.sceneBundle.scene, this.sceneBundle.camera);
  }

  public dispose() {
    this.running = false;
    for (const fn of this.cleanups) fn();
    this.cleanups = [];
    this.sceneBundle.controls.dispose();
    this.baseMesh.geometry.dispose();
    (this.baseMesh.material as THREE.Material).dispose();
    this.crossSectionMesh.geometry.dispose();
    (this.crossSectionMesh.material as THREE.Material).dispose();
    this.curveRenderer.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    if (this.scalarsND) { this.scalarsND.delete(); this.scalarsND = null; }
    if (this.normalVec) { this.normalVec.delete(); this.normalVec = null; }
    this.mesh.delete();
  }
}

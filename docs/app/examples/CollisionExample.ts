import { fitCameraToAllMeshesFromZPlane, createScene, type SceneBundle } from "@/utils/sceneUtils";
import { centerAndScale, pickMesh, randomTransformation, RollingAverage } from "@/utils/utils";
import * as THREE from "three";

type TF = typeof import("@polydera/trueform");

export class CollisionExample {
  private tf: TF;
  private baseMesh: any;
  private tfMeshes: any[] = [];
  private container: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private sceneBundle: SceneBundle;
  private threeMeshes: THREE.Mesh[] = [];
  private running = true;
  private cleanups: (() => void)[] = [];

  private selectedId: number | null = null;
  private dragging = false;
  private movingPlane = new THREE.Plane();
  private lastPoint = new THREE.Vector3();
  private colliding = new Set<number>();
  private pickTiming = new RollingAverage();
  private collideTiming = new RollingAverage();

  private raycaster = new THREE.Raycaster();
  private ndc = new THREE.Vector2();

  private normalColor = new THREE.Color(0.8, 0.8, 0.8);
  private collidingColor = new THREE.Color(0.7, 1, 1);

  public refreshTimeValue: (() => number) | null = null;

  constructor(tf: TF, fileBuffer: ArrayBuffer, fileName: string, container: HTMLElement, isDarkMode = true) {
    this.tf = tf;
    this.container = container;

    // Load mesh
    const ext = fileName.split(".").pop()?.toLowerCase();
    this.baseMesh = ext === "stl" ? tf.readStl(fileBuffer) : tf.readObj(fileBuffer);
    centerAndScale(tf, this.baseMesh);

    // Create 5x5 grid of meshes: baseMesh for first slot, shallowCopy() for the rest
    const gridSize = 5;
    const spacing = 15;
    for (let j = 0; j < gridSize; j++) {
      for (let i = 0; i < gridSize; i++) {
        const tfMesh = (i === 0 && j === 0) ? this.baseMesh : this.baseMesh.shallowCopy();
        const mat = randomTransformation(tf, i * spacing, j * spacing, 0);
        tfMesh.transformation = mat;
        mat.delete();
        this.tfMeshes.push(tfMesh);
      }
    }

    // Three.js setup
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    container.appendChild(this.renderer.domElement);

    this.sceneBundle = createScene(this.renderer, {
      backgroundColor: isDarkMode ? 0x1e1e1e : 0xfafafa,
      enableFog: false,
    });

    // Shared geometry from base mesh points/faces
    const points = this.baseMesh.points;
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

    // Create 25 Three.js meshes, each with own material (for independent color), sharing geometry
    const materials: THREE.MeshMatcapMaterial[] = [];
    for (let idx = 0; idx < this.tfMeshes.length; idx++) {
      const material = new THREE.MeshMatcapMaterial({
        side: THREE.DoubleSide,
        flatShading: true,
        color: this.normalColor.clone(),
      });
      materials.push(material);
      const mesh = new THREE.Mesh(sharedGeometry, material);
      mesh.matrixAutoUpdate = false;
      this.syncThreeMatrix(idx, mesh);
      this.sceneBundle.scene.add(mesh);
      this.threeMeshes.push(mesh);
    }

    // Load matcap texture and assign to all materials
    new THREE.TextureLoader().load(matcapUrl, (tex) => {
      for (let i = 0; i < materials.length; i++) {
        materials[i].matcap = i === 0 ? tex : tex.clone();
        materials[i].needsUpdate = true;
      }
    });

    // Fit camera
    fitCameraToAllMeshesFromZPlane(this.sceneBundle, 1.5);

    // Pointer events
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
      this.handleHover(); // pick on click
      if (this.selectedId !== null) {
        this.dragging = true;
        this.sceneBundle.controls.enabled = false;
      }
    };
    const onPointerUp = () => {
      this.dragging = false;
      this.selectedId = null;
      this.sceneBundle.controls.enabled = true;
      this.colliding.clear();
      this.updateColors();
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

    const t0 = performance.now();
    const hit = pickMesh(tf, tfRay, this.tfMeshes);
    this.pickTiming.add(performance.now() - t0);
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

    if (this.refreshTimeValue) this.refreshTimeValue();
  }

  private handleDrag() {
    const tf = this.tf;
    const id = this.selectedId!;

    this.raycaster.setFromCamera(this.ndc, this.sceneBundle.camera);
    const nextPoint = new THREE.Vector3();
    this.raycaster.ray.intersectPlane(this.movingPlane, nextPoint);
    if (!nextPoint) return;

    const dx = nextPoint.x - this.lastPoint.x;
    const dy = nextPoint.y - this.lastPoint.y;
    const dz = nextPoint.z - this.lastPoint.z;
    this.lastPoint.copy(nextPoint);

    // Update tf mesh transformation: read, modify translation in-place, write back
    const mat = this.tfMeshes[id].transformation;
    const d = mat.data; // mutable WASM heap view
    d[3] += dx; d[7] += dy; d[11] += dz;
    this.tfMeshes[id].transformation = mat;
    mat.delete();

    // Sync Three.js matrix
    this.syncThreeMatrix(id);

    // Collision detection
    const t0 = performance.now();
    this.colliding.clear();
    for (let i = 0; i < this.tfMeshes.length; i++) {
      if (i === id) continue;
      if (tf.intersects(this.tfMeshes[id], this.tfMeshes[i])) {
        this.colliding.add(i);
      }
    }
    this.collideTiming.add(performance.now() - t0);

    this.updateColors();
    if (this.refreshTimeValue) this.refreshTimeValue();
  }

  private syncThreeMatrix(index: number, mesh?: THREE.Mesh) {
    const target = mesh ?? this.threeMeshes[index];
    const mat = this.tfMeshes[index].transformation;
    if (!mat) return;
    const m = new THREE.Matrix4();
    m.fromArray(mat.data);
    m.transpose(); // row-major → column-major
    target.matrix.copy(m);
    mat.delete();
  }

  private updateColors() {
    for (let i = 0; i < this.threeMeshes.length; i++) {
      const mat = this.threeMeshes[i].material as THREE.MeshMatcapMaterial;
      mat.color.copy(this.colliding.has(i) ? this.collidingColor : this.normalColor);
    }
  }

  public getAverageTime(): number {
    return this.collideTiming.average;
  }

  public getAveragePickTime(): number {
    return this.pickTiming.average;
  }

  public applyTheme(isDark: boolean) {
    this.sceneBundle.scene.background = new THREE.Color(isDark ? 0x1e1e1e : 0xfafafa);
    const matcapUrl = isDark
      ? "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/635D52_A9BCC0_B1AEA0_819598.png"
      : "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/2D2D2F_C6C2C5_727176_94949B.png";
    new THREE.TextureLoader().load(matcapUrl, (tex) => {
      for (let i = 0; i < this.threeMeshes.length; i++) {
        const mat = this.threeMeshes[i].material as THREE.MeshMatcapMaterial;
        if (mat.matcap) mat.matcap.dispose();
        mat.matcap = i === 0 ? tex : tex.clone();
        mat.needsUpdate = true;
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
    this.renderer.dispose();
    this.renderer.domElement.remove();
    // Delete tf meshes (shallow copies first, then baseMesh)
    for (let i = this.tfMeshes.length - 1; i >= 0; i--) {
      this.tfMeshes[i].delete();
    }
    this.tfMeshes = [];
  }
}

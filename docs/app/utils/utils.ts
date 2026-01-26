import * as THREE from "three";
import { SRGBColorSpace } from "three";
import type { instance, mesh_data, result_mesh } from "@/examples/native";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

// ============================================================================
// CurveRenderer - Efficient instanced tube rendering (cylinders + spheres)
// ============================================================================

export interface CurveRendererOpts {
  radius?: number;
  color?: number;
  maxSegments?: number;
  cylinderRadialSegments?: number;
  sphereWidthSegments?: number;
  sphereHeightSegments?: number;
}

export class CurveRenderer {
  public readonly object: THREE.Group;

  private cylinderMesh: THREE.InstancedMesh;
  private sphereMesh: THREE.InstancedMesh;
  private cylinderBuffer: Float32Array;
  private sphereBuffer: Float32Array;
  private maxSegments: number;
  private radius: number;

  constructor(opts: CurveRendererOpts = {}) {
    const {
      radius = 0.12,
      color = 0xff2020,
      maxSegments = 20000,
      cylinderRadialSegments = 6,
      sphereWidthSegments = 8,
      sphereHeightSegments = 6,
    } = opts;

    this.radius = radius;
    this.maxSegments = maxSegments;
    this.object = new THREE.Group();
    this.object.name = "curve_renderer";

    // Shared material
    const material = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.4,
      metalness: 0.1,
    });

    // Cylinder: unit height along Y, will transform per-instance
    const cylGeom = new THREE.CylinderGeometry(1, 1, 1, cylinderRadialSegments);
    this.cylinderMesh = new THREE.InstancedMesh(cylGeom, material, maxSegments);
    this.cylinderMesh.count = 0;
    this.cylinderMesh.frustumCulled = false;
    this.cylinderBuffer = this.cylinderMesh.instanceMatrix.array as Float32Array;

    // Sphere: unit radius, will scale per-instance
    const sphereGeom = new THREE.SphereGeometry(1, sphereWidthSegments, sphereHeightSegments);
    this.sphereMesh = new THREE.InstancedMesh(sphereGeom, material.clone(), maxSegments);
    this.sphereMesh.count = 0;
    this.sphereMesh.frustumCulled = false;
    this.sphereBuffer = this.sphereMesh.instanceMatrix.array as Float32Array;

    this.object.add(this.cylinderMesh);
    this.object.add(this.sphereMesh);
  }

  /**
   * Update curve rendering. Call each frame when curves change.
   */
  update(curves: { points: Float32Array; paths: number[][] }): void {
    const { points, paths } = curves;

    if (!points || !paths || points.length === 0) {
      this.cylinderMesh.count = 0;
      this.sphereMesh.count = 0;
      return;
    }

    const vertCount = points.length / 3;
    const r = this.radius;

    let cylIdx = 0;
    let sphIdx = 0;

    for (const path of paths) {
      if (!path || path.length < 2) continue;

      for (let i = 0; i < path.length; i++) {
        const idx = path[i];
        if (idx < 0 || idx >= vertCount) continue;

        const x = points[idx * 3];
        const y = points[idx * 3 + 1];
        const z = points[idx * 3 + 2];

        // Sphere at this vertex
        if (sphIdx < this.maxSegments) {
          this.writeSphereMatrix(sphIdx, x, y, z, r);
          sphIdx++;
        }

        // Cylinder to next vertex
        if (i < path.length - 1) {
          const nextIdx = path[i + 1];
          if (nextIdx < 0 || nextIdx >= vertCount) continue;

          const nx = points[nextIdx * 3];
          const ny = points[nextIdx * 3 + 1];
          const nz = points[nextIdx * 3 + 2];

          if (cylIdx < this.maxSegments) {
            this.writeCylinderMatrix(cylIdx, x, y, z, nx, ny, nz, r);
            cylIdx++;
          }
        }
      }
    }

    this.cylinderMesh.count = cylIdx;
    this.sphereMesh.count = sphIdx;
    this.cylinderMesh.instanceMatrix.needsUpdate = true;
    this.sphereMesh.instanceMatrix.needsUpdate = true;
  }

  /**
   * Write sphere matrix directly to buffer (identity rotation, uniform scale, translation)
   */
  private writeSphereMatrix(idx: number, x: number, y: number, z: number, radius: number): void {
    const o = idx * 16;
    const buf = this.sphereBuffer;
    // Column-major 4x4: scale on diagonal, translation in column 3
    buf[o + 0] = radius;
    buf[o + 1] = 0;
    buf[o + 2] = 0;
    buf[o + 3] = 0;
    buf[o + 4] = 0;
    buf[o + 5] = radius;
    buf[o + 6] = 0;
    buf[o + 7] = 0;
    buf[o + 8] = 0;
    buf[o + 9] = 0;
    buf[o + 10] = radius;
    buf[o + 11] = 0;
    buf[o + 12] = x;
    buf[o + 13] = y;
    buf[o + 14] = z;
    buf[o + 15] = 1;
  }

  /**
   * Write cylinder matrix directly to buffer (rotation to align with segment, scale by length)
   */
  private writeCylinderMatrix(
    idx: number,
    x0: number,
    y0: number,
    z0: number,
    x1: number,
    y1: number,
    z1: number,
    radius: number,
  ): void {
    const o = idx * 16;
    const buf = this.cylinderBuffer;

    // Direction and length
    const dx = x1 - x0;
    const dy = y1 - y0;
    const dz = z1 - z0;
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);

    if (len < 1e-8) {
      // Degenerate - write identity with zero scale
      buf[o + 0] = 0;
      buf[o + 1] = 0;
      buf[o + 2] = 0;
      buf[o + 3] = 0;
      buf[o + 4] = 0;
      buf[o + 5] = 0;
      buf[o + 6] = 0;
      buf[o + 7] = 0;
      buf[o + 8] = 0;
      buf[o + 9] = 0;
      buf[o + 10] = 0;
      buf[o + 11] = 0;
      buf[o + 12] = x0;
      buf[o + 13] = y0;
      buf[o + 14] = z0;
      buf[o + 15] = 1;
      return;
    }

    // Normalized direction (this will be the Y axis of the cylinder)
    const ax = dx / len;
    const ay = dy / len;
    const az = dz / len;

    // Find perpendicular vectors for X and Z axes
    // Pick a non-parallel reference vector
    let rx: number, ry: number, rz: number;
    if (Math.abs(ay) < 0.9) {
      rx = 0;
      ry = 1;
      rz = 0;
    } else {
      rx = 1;
      ry = 0;
      rz = 0;
    }

    // X axis = cross(Y, ref), normalized
    let bx = ay * rz - az * ry;
    let by = az * rx - ax * rz;
    let bz = ax * ry - ay * rx;
    const bLen = Math.sqrt(bx * bx + by * by + bz * bz);
    bx /= bLen;
    by /= bLen;
    bz /= bLen;

    // Z axis = cross(X, Y)
    const cx = by * az - bz * ay;
    const cy = bz * ax - bx * az;
    const cz = bx * ay - by * ax;

    // Midpoint
    const mx = (x0 + x1) * 0.5;
    const my = (y0 + y1) * 0.5;
    const mz = (z0 + z1) * 0.5;

    // Column-major 4x4 with rotation and non-uniform scale
    // Column 0: X axis * radius
    buf[o + 0] = bx * radius;
    buf[o + 1] = by * radius;
    buf[o + 2] = bz * radius;
    buf[o + 3] = 0;
    // Column 1: Y axis * length (cylinder height)
    buf[o + 4] = ax * len;
    buf[o + 5] = ay * len;
    buf[o + 6] = az * len;
    buf[o + 7] = 0;
    // Column 2: Z axis * radius
    buf[o + 8] = cx * radius;
    buf[o + 9] = cy * radius;
    buf[o + 10] = cz * radius;
    buf[o + 11] = 0;
    // Column 3: translation (midpoint)
    buf[o + 12] = mx;
    buf[o + 13] = my;
    buf[o + 14] = mz;
    buf[o + 15] = 1;
  }

  dispose(): void {
    this.cylinderMesh.geometry.dispose();
    this.sphereMesh.geometry.dispose();
    (this.cylinderMesh.material as THREE.Material).dispose();
    (this.sphereMesh.material as THREE.Material).dispose();
    this.object.clear();
  }
}

export interface CurveObj {
  points: Float32Array;
  paths: number[][];
}

export interface CurveLineObjects {
  lines: LineSegments2;
}

export interface curvesToCurvePolyOpts {
  radius?: number;
  radialSegments?: number;
  tubularSegPerEdge?: number;
  tubeColor?: number;
  lineWidth?: number;
  alwaysOnTop?: boolean; // Whether lines should always render on top of other objects
  renderOrder?: number; // Custom render order for fine-grained control
}

export function createPoints() {
  const pointPixelSize = 1.0;
  const pointColor = 0xff00ff;
  const pointGeom = new THREE.BufferGeometry();
  const pointMat = new THREE.PointsMaterial({
    color: pointColor,
    size: pointPixelSize,
    sizeAttenuation: true,
  });
  const pointsObj = new THREE.Points(pointGeom, pointMat);
  pointsObj.name = "curve_points";
  pointsObj.matrixAutoUpdate = false;
  return pointsObj;
}

export function switchTextures(mesh: THREE.Mesh, isDarkMode: boolean) {
  if (!mesh.material || !(mesh.material instanceof THREE.MeshMatcapMaterial)) return;
  const material = mesh.material as THREE.MeshMatcapMaterial;
  if (!material.matcap) return;
  material.matcap.dispose();
  let path: string;
  if (isDarkMode) {
    path =
      "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/635D52_A9BCC0_B1AEA0_819598.png";
  } else {
    path =
      "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/2D2D2F_C6C2C5_727176_94949B.png";
  }
  material.matcap = new THREE.TextureLoader().load(path);
}
/**
 * Create a matcap material for instanced meshes.
 * Returns a material suitable for THREE.InstancedMesh with per-instance colors.
 */
export function createInstancedMaterial(isDarkMode: boolean) {
  let path: string;
  if (isDarkMode) {
    path =
      "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/635D52_A9BCC0_B1AEA0_819598.png";
  } else {
    path =
      "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/2D2D2F_C6C2C5_727176_94949B.png";
  }
  const matcapTexture = new THREE.TextureLoader().load(path);
  const material = new THREE.MeshMatcapMaterial({
    matcap: matcapTexture,
    side: THREE.DoubleSide,
    flatShading: true,
  });
  return material;
}

export function createMesh(isDarkMode: boolean) {
  // TODO test and choose here: https://observablehq.com/@makio135/matcaps
  // "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/2D2D2F_C6C2C5_727176_94949B.png"
  // "https://makio135.com/matcaps/1024/7877EE_D87FC5_75D9C7_1C78C0.png"
  // "https://raw.githubusercontent.com/nidorx/matcaps/master/512/7C584C_27140D_B3765C_3D2318-512px.png"
  // "https://makio135.com/matcaps/64/593E2C_E5D8A9_BC9F79_9F8A68-64px.png"
  // "https://raw.githubusercontent.com/nidorx/matcaps/master/512/815C41_F6C99A_D39F77_BB9474-512px.png"
  // "https://raw.githubusercontent.com/nidorx/matcaps/master/512/AD9E81_F1E5CE_6B5C3E_5A492A-512px.png"
  // "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/AE9D99_29303B_585F70_875C33.png"
  // "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/635D52_A9BCC0_B1AEA0_819598.png",
  let path: string;
  if (isDarkMode) {
    path =
      "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/635D52_A9BCC0_B1AEA0_819598.png";
  } else {
    path =
      "https://raw.githubusercontent.com/nidorx/matcaps/master/1024/2D2D2F_C6C2C5_727176_94949B.png";
  }
  const matcapTexture = new THREE.TextureLoader().load(path);
  const material = new THREE.MeshMatcapMaterial({
    matcap: matcapTexture,
    side: THREE.DoubleSide,
    flatShading: true,
  });

  const geometry = new THREE.BufferGeometry();
  const mesh = new THREE.Mesh(geometry, material);
  mesh.matrixAutoUpdate = false;
  return mesh;
}

/**
 * Initialize mesh geometry from mesh_data (called once per unique mesh)
 */
export function initMeshGeometry(data: mesh_data, mesh: THREE.Mesh) {
  const geometry = mesh.geometry;
  const positions = new Float32Array(data.get_points());
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(data.get_faces()), 1));
  if (positions.length >= 3) {
    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();
  }
}

/**
 * Update mesh transform and color from instance (called per frame)
 */
export function updateMeshFromInstance(inst: instance, mesh: THREE.Mesh) {
  // Update color
  const currC = (mesh.material as THREE.MeshMatcapMaterial).color;
  const newC = inst.color;
  if (currC.r !== newC[0] || currC.g !== newC[1] || currC.b !== newC[2]) {
    currC.setRGB(newC[0], newC[1], newC[2], SRGBColorSpace);
  }
  // Update matrix
  if (inst.matrix_updated) {
    const matrix = new Float32Array(inst.get_matrix());
    const threeMatrix = new THREE.Matrix4();
    threeMatrix.fromArray(matrix);
    threeMatrix.transpose();
    mesh.matrix = threeMatrix;
  }
}

/**
 * Update result mesh geometry (for boolean results, isobands, etc.)
 */
export function updateResultMesh(result: result_mesh, mesh: THREE.Mesh) {
  if (result.updated) {
    const geometry = mesh.geometry;
    const positions = new Float32Array(result.get_points());
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(result.get_faces()), 1));
    if (positions.length >= 3) {
      geometry.computeBoundingSphere();
      geometry.computeBoundingBox();
    }
  }
}

// Legacy compatibility - combines mesh_data and instance for migration
export function getMeshFromWasm(
  data: mesh_data,
  inst: instance,
  mesh: THREE.Mesh,
  needsGeometryInit: boolean,
) {
  if (needsGeometryInit) {
    initMeshGeometry(data, mesh);
  }
  updateMeshFromInstance(inst, mesh);
}

export function getMatrixFromWasm(
  inst: instance,
  dstGeometry: THREE.Mesh | THREE.Line | THREE.Points,
) {
  const mU = inst.matrix_updated;
  if (mU) {
    const matrix = new Float32Array(inst.get_matrix());
    const threeMatrix = new THREE.Matrix4();
    threeMatrix.fromArray(matrix);
    threeMatrix.transpose();
    dstGeometry.matrix = threeMatrix;
  }
}

/**
 * Convert flat buffers to the {points, paths} structure used by curvesToCurvePoly().
 *
 * @param {Float32Array|number[]} pointsFlat  [x0,y0,z0, x1,y1,z1, ...]
 * @param {Int32Array|Uint32Array|number[]} ids
 * @param {Int32Array|Uint32Array|number[]} offsets  [0, a, b, ..., ids.length]
 * @returns CurveObj
 */
export function buffersToCurves(points: Float32Array, idBuf: Int32Array, offBuf: Int32Array) {
  // basic validation
  if (points.length % 3 !== 0) {
    console.warn("points length must be multiple of 3");
    return { points: new Float32Array(), paths: [] };
  }
  if (offBuf.length < 2) {
    console.warn("offsets must have at least [0, ids.length]");
    return { points: new Float32Array(), paths: [] };
  }
  if (offBuf[0] !== 0 || offBuf[offBuf.length - 1] !== idBuf.length) {
    console.warn("offsets must start at 0 and end at ids.length");
    return { points: new Float32Array(), paths: [] };
  }

  const nPaths = offBuf.length - 1;
  const paths = new Array(nPaths);

  for (let p = 0; p < nPaths; p++) {
    const start = offBuf[p];
    const end = offBuf[p + 1];
    const len = Math.max(0, end - start);

    const path = new Array(len);
    for (let i = 0; i < len; i++) {
      path[i] = idBuf[start + i];
    }
    paths[p] = path;
  }
  return { points, paths };
}

/**
 * Creates reusable curve line objects that can be updated instead of recreated.
 * This is much more efficient than TubeGeometry for performance-critical applications.
 *
 * @param {curvesToCurvePolyOpts} [opts] Configuration options
 * @returns {CurveLineObjects} Reusable objects for line-based curve visualization
 */
export function createCurveLineObjects(opts: curvesToCurvePolyOpts = {}): CurveLineObjects {
  const {
    tubeColor = 0xff00ff,
    lineWidth = 2.0,
    alwaysOnTop = false,
    renderOrder = 1, // High render order by default
  } = opts;

  // Create line segments object
  const lineGeom = new LineSegmentsGeometry();
  const lineMat = new LineMaterial({
    color: tubeColor,
    linewidth: lineWidth, // in worldUnit
    worldUnits: true,
    vertexColors: false,
    alphaToCoverage: true,
    transparent: true,
    opacity: 0.7,
    depthTest: !alwaysOnTop,
    depthWrite: !alwaysOnTop,

    // --- depth bias to pull the line toward the camera ---
    polygonOffset: true,
    polygonOffsetFactor: -10,
    polygonOffsetUnits: -2,

    side: THREE.DoubleSide,
  });
  const linesObj = new LineSegments2(lineGeom, lineMat);
  linesObj.name = "curve_lines";
  linesObj.visible = true;
  linesObj.renderOrder = renderOrder;

  return { lines: linesObj };
}
/**
 * Fast smoothing + update for MANY paths each frame.
 * - paths: number[][] (each entry is a path of vertex indices)
 * - Chaikin smoothing (0..3 iterations)
 * - One allocation per frame, one setPositions() upload
 */
export function curvesToCurveLinesFast(
  curves: { points: Float32Array; paths: number[][] },
  curveObjects: { lines: any }, // LineSegments2
  opts?: { chaikinIterations?: number; closed?: boolean },
) {
  const t0 = performance.now();

  const { points: src, paths } = curves;
  if (!src || !paths) throw new Error("curves must have {points, paths}");

  const segsPerPath: number[] = [];
  const iterations = Math.max(0, Math.min(3, opts?.chaikinIterations ?? 2));
  const closed = !!opts?.closed;
  const lineGeom = curveObjects.lines.geometry as any; // LineSegmentsGeometry
  const vertCount = src.length / 3;

  const EPS = 1e-12;
  let skippedDegenerate = 0,
    skippedNaN = 0;
  function pushSeg(x0: number, y0: number, z0: number, x1: number, y1: number, z1: number) {
    if (
      !Number.isFinite(x0) ||
      !Number.isFinite(y0) ||
      !Number.isFinite(z0) ||
      !Number.isFinite(x1) ||
      !Number.isFinite(y1) ||
      !Number.isFinite(z1)
    ) {
      skippedNaN++;
      return false;
    }
    const dx = x1 - x0,
      dy = y1 - y0,
      dz = z1 - z0;
    if (dx * dx + dy * dy + dz * dz < EPS) {
      skippedDegenerate++;
      return false;
    }
    work.flat[w++] = x0;
    work.flat[w++] = y0;
    work.flat[w++] = z0;
    work.flat[w++] = x1;
    work.flat[w++] = y1;
    work.flat[w++] = z1;
    return true;
  }

  // workspace stored on the mesh for reuse
  type Work = {
    ax: Float32Array;
    ay: Float32Array;
    az: Float32Array; // stage A
    bx: Float32Array;
    by: Float32Array;
    bz: Float32Array; // stage B
    flat: Float32Array; // [a,b,a,b,...]
  };
  const work: Work = (curveObjects.lines.userData.__work ??= {
    ax: new Float32Array(0),
    ay: new Float32Array(0),
    az: new Float32Array(0),
    bx: new Float32Array(0),
    by: new Float32Array(0),
    bz: new Float32Array(0),
    flat: new Float32Array(0),
  });

  const nextPow2 = (n: number) => 1 << Math.ceil(Math.log2(Math.max(1, n)));
  const ensureLen = (arr: Float32Array, needed: number) =>
    arr.length >= needed ? arr : new Float32Array(nextPow2(needed));

  // ---- PASS 1: count total segments this frame (so we allocate once) --------
  let totalSegments = 0;
  for (let p = 0; p < paths.length; p++) {
    const idx = paths[p];
    if (!idx) continue;

    // count valid points in this path
    let L = 0;
    for (let i = 0; i < idx.length; i++) {
      const vi = idx[i];
      if (vi != null && vi >= 0 && vi < vertCount) L++;
    }
    if (L < 2) continue;

    // points after k Chaikin iterations
    // open:  (L - 1)*2^k + 1        -> segments = (L - 1)*2^k
    // closed: L*2^k                  -> segments =  L*2^k
    const pow2k = 1 << iterations;
    totalSegments += closed ? L * pow2k : (L - 1) * pow2k;
  }

  // ensure the big flat buffer fits ALL paths
  const neededFloats = totalSegments * 6; // a.xyz + b.xyz per segment
  if (work.flat.length < neededFloats) {
    work.flat = new Float32Array(nextPow2(neededFloats));
  }

  // ---- PASS 2: build all segments into the flat buffer ----------------------
  let w = 0; // write cursor in floats

  for (let p = 0; p < paths.length; p++) {
    const idx = paths[p];
    if (!idx) continue;

    // stage A: copy valid points of this path
    // size A large enough for the raw path
    work.ax = ensureLen(work.ax, idx.length);
    work.ay = ensureLen(work.ay, idx.length);
    work.az = ensureLen(work.az, idx.length);

    let n = 0;
    for (let i = 0; i < idx.length; i++) {
      const vi = idx[i];
      if (vi == null || vi < 0 || vi >= vertCount) continue;
      const j = 3 * vi;
      work.ax[n] = src[j];
      work.ay[n] = src[j + 1];
      work.az[n] = src[j + 2];
      n++;
    }
    if (n < 2) continue;

    // Chaikin smoothing
    for (let it = 0; it < iterations; it++) {
      const newN = closed ? n * 2 : (n - 1) * 2 + 1;
      work.bx = ensureLen(work.bx, newN);
      work.by = ensureLen(work.by, newN);
      work.bz = ensureLen(work.bz, newN);

      let k = 0;
      if (!closed) {
        work.bx[k] = work.ax[0];
        work.by[k] = work.ay[0];
        work.bz[k] = work.az[0];
        k++;
        for (let i = 0; i < n - 1; i++) {
          const x0 = work.ax[i],
            y0 = work.ay[i],
            z0 = work.az[i];
          const x1 = work.ax[i + 1],
            y1 = work.ay[i + 1],
            z1 = work.az[i + 1];
          work.bx[k] = 0.75 * x0 + 0.25 * x1;
          work.by[k] = 0.75 * y0 + 0.25 * y1;
          work.bz[k++] = 0.75 * z0 + 0.25 * z1;
          work.bx[k] = 0.25 * x0 + 0.75 * x1;
          work.by[k] = 0.25 * y0 + 0.75 * y1;
          work.bz[k++] = 0.25 * z0 + 0.75 * z1;
        }
        work.bx[k] = work.ax[n - 1];
        work.by[k] = work.ay[n - 1];
        work.bz[k] = work.az[n - 1];
        k++;
      } else {
        for (let i = 0; i < n; i++) {
          const ni = (i + 1) % n;
          const x0 = work.ax[i],
            y0 = work.ay[i],
            z0 = work.az[i];
          const x1 = work.ax[ni],
            y1 = work.ay[ni],
            z1 = work.az[ni];
          work.bx[k] = 0.75 * x0 + 0.25 * x1;
          work.by[k] = 0.75 * y0 + 0.25 * y1;
          work.bz[k++] = 0.75 * z0 + 0.25 * z1;
          work.bx[k] = 0.25 * x0 + 0.75 * x1;
          work.by[k] = 0.25 * y0 + 0.75 * y1;
          work.bz[k++] = 0.25 * z0 + 0.75 * z1;
        }
      }

      // swap B -> A
      [work.ax, work.ay, work.az, work.bx, work.by, work.bz] = [
        work.bx,
        work.by,
        work.bz,
        work.ax,
        work.ay,
        work.az,
      ];
      n = k;
    }

    // Emit this path's segments into the global flat buffer
    if (!closed) {
      for (let i = 0; i < n - 1; i++) {
        pushSeg(work.ax[i], work.ay[i], work.az[i], work.ax[i + 1], work.ay[i + 1], work.az[i + 1]);
        // work.flat[w++] = work.ax[i];
        // work.flat[w++] = work.ay[i];
        // work.flat[w++] = work.az[i];
        // work.flat[w++] = work.ax[i+1];
        // work.flat[w++] = work.ay[i+1];
        // work.flat[w++] = work.az[i+1];
      }
    } else {
      for (let i = 0; i < n; i++) {
        const ni = (i + 1) % n;
        pushSeg(work.ax[i], work.ay[i], work.az[i], work.ax[ni], work.ay[ni], work.az[ni]);
        // work.flat[w++] = work.ax[i];
        // work.flat[w++] = work.ay[i];
        // work.flat[w++] = work.az[i];
        // work.flat[w++] = work.ax[ni];
        // work.flat[w++] = work.ay[ni];
        // work.flat[w++] = work.az[ni];
      }
    }

    segsPerPath.push(n);
  }

  const usedFloats = w;
  const usedSegments = usedFloats / 6;

  // Upload once; LineSegmentsGeometry will set instanceStart/End etc.
  if (usedSegments > 0) {
    lineGeom.setPositions(work.flat.subarray(0, usedFloats));
    const segs = usedFloats / 6; // total segments you wrote
    (lineGeom as any).instanceCount = segs; // r152+
    (lineGeom as any).maxInstancedCount = segs; // r14x–r15x
    const aS = lineGeom.getAttribute("instanceStart") as THREE.InstancedBufferAttribute;
    const aE = lineGeom.getAttribute("instanceEnd") as THREE.InstancedBufferAttribute;
    function logInstance(i: number) {
      const s = [aS.getX(i), aS.getY(i), aS.getZ(i)];
      const e = [aE.getX(i), aE.getY(i), aE.getZ(i)];
      console.log(
        "inst",
        i,
        "start",
        s,
        "end",
        e,
        "len",
        Math.hypot(e[0] - s[0], e[1] - s[1], e[2] - s[2]),
      );
    }
    logInstance(0);
    logInstance(Math.floor(aS.count / 3)); // middle
    logInstance(aS.count - 1); // last

    // Bounds (over used range) — robust for all r14x–r16x
    {
      let minX = Infinity,
        minY = Infinity,
        minZ = Infinity;
      let maxX = -Infinity,
        maxY = -Infinity,
        maxZ = -Infinity;
      for (let i = 0; i < usedFloats; i += 3) {
        const x = work.flat[i],
          y = work.flat[i + 1],
          z = work.flat[i + 2];
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
      }
      const box = new THREE.Box3(
        new THREE.Vector3(minX, minY, minZ),
        new THREE.Vector3(maxX, maxY, maxZ),
      );
      lineGeom.boundingBox = box;

      const center = box.getCenter(new THREE.Vector3());
      let r2 = 0;
      for (let i = 0; i < usedFloats; i += 3) {
        const dx = work.flat[i] - center.x;
        const dy = work.flat[i + 1] - center.y;
        const dz = work.flat[i + 2] - center.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > r2) r2 = d2;
      }
      lineGeom.boundingSphere = new THREE.Sphere(center, Math.sqrt(r2));
    }

    // (lineGeom as any).instanceCount = usedSegments; // harmless; setPositions also handles this
    curveObjects.lines.visible = true;
    curveObjects.lines.frustumCulled = false;
  } else {
    lineGeom.setPositions(new Float32Array(0));
    curveObjects.lines.visible = false;
  }

  console.log(
    `curvesToCurveLinesFast: ${(performance.now() - t0) | 0} ms, paths=${paths.length}, segments=${usedSegments}`,
  );
  const aS = lineGeom.getAttribute("instanceStart");
  const aE = lineGeom.getAttribute("instanceEnd");

  console.log({
    cpuSegments: usedFloats / 6,
    gpu_instanceCount: (lineGeom as any).instanceCount,
    gpu_maxInstancedCount: (lineGeom as any).maxInstancedCount,
    attr_instanceStart_count: aS?.count,
    attr_instanceEnd_count: aE?.count,
    base_position_count: lineGeom.getAttribute("position")?.count, // should be 6
  });
  console.log("base_position_count", lineGeom.getAttribute("position")?.count);
  console.table(segsPerPath.map((n, i) => ({ path: i, segs: n })));
  console.log("skipped degenerate:", skippedDegenerate, "skipped NaN:", skippedNaN);

  return curveObjects;
}

/**
 * Smooths each curve path with a centripetal Catmull-Rom spline and
 * rebuilds the LineSegmentsGeometry from the resampled points.
 *
 * Tunables:
 *  - samplesPerSegment: how many samples to add between original points
 *  - tension: Catmull-Rom tension (0.0..1.0). 0.5 is a good default.
 *  - closed: set true if your paths are closed loops
 */
export function curvesToCurveLines(
  curves: { points: Float32Array; paths: number[][] },
  curveObjects: { lines: THREE.LineSegments },
  opts?: { samplesPerSegment?: number; tension?: number; closed?: boolean },
) {
  const t0 = performance.now();

  const { points: srcPoints, paths } = curves;
  if (!srcPoints || !paths) throw new Error("curves must have {points, paths}");

  const samplesPerSegment = opts?.samplesPerSegment ?? 6; // ↑ for smoother
  const tension = opts?.tension ?? 0.5; // centripetal feel
  const closedDefault = opts?.closed ?? true;

  // helper to read a point by index from the flat Float32Array
  const getV = (i: number) =>
    new THREE.Vector3(srcPoints[3 * i], srcPoints[3 * i + 1], srcPoints[3 * i + 2]);

  // --- build smoothed line segments ---
  const linePositions: number[] = [];

  for (const path of paths) {
    if (!path || path.length < 2) continue;

    // original points for this path
    const pts: THREE.Vector3[] = path.map(getV);

    // spline through them
    const curve = new THREE.CatmullRomCurve3(
      pts,
      /*closed=*/ closedDefault,
      /*type=*/ "centripetal",
      /*tension=*/ tension,
    );

    // resample: original count + (N * (#segments))
    const samples = Math.max(2, (path.length - 1) * samplesPerSegment + 1);
    const smooth: THREE.Vector3[] = curve.getPoints(samples);

    // emit segments between successive resampled points
    for (let i = 0; i < smooth.length - 1; i++) {
      const a = smooth[i],
        b = smooth[i + 1];
      linePositions.push(a.x, a.y, a.z, b.x, b.y, b.z);
    }
  }

  // --- swap in a fresh LineSegmentsGeometry ---
  const oldGeom = curveObjects.lines.geometry as LineSegmentsGeometry;
  const newGeom = new LineSegmentsGeometry();

  if (linePositions.length > 0) {
    // LineSegmentsGeometry accepts a flat array of xyz pairs
    newGeom.setPositions(linePositions);
  }

  oldGeom.dispose();
  curveObjects.lines.geometry = newGeom;

  // console.log("Time to build line segments: " + (performance.now() - t0) + "ms");
  return curveObjects;
}
/**
 * Alternative implementation using basic THREE.LineSegments for compatibility.
 * Use this if LineSegments2 is not working properly.
 */
export function createBasicCurveLineObjects(opts: curvesToCurvePolyOpts = {}): any {
  const { tubeColor = 0xff00ff, lineWidth = 2.0 } = opts;

  // Create basic line segments object
  const lineGeom = new THREE.BufferGeometry();
  const lineMat = new THREE.LineBasicMaterial({
    color: tubeColor,
    linewidth: lineWidth, // Note: linewidth only works on Windows
  });
  const linesObj = new THREE.LineSegments(lineGeom, lineMat);
  linesObj.name = "curve_lines";
  linesObj.visible = true;

  return { lines: linesObj };
}

/**
 * Updates basic LineSegments geometry (for use with createBasicCurveLineObjects)
 */
export function updateBasicCurveLines(curves: CurveObj, curveObjects: any): any {
  const { points: srcPoints, paths } = curves;
  if (!srcPoints || !paths) throw new Error("curves must have {points, paths}");

  // Build line segments for each polyline path
  const linePositions = [];

  const getV = (i: number) => [srcPoints[3 * i], srcPoints[3 * i + 1], srcPoints[3 * i + 2]];

  for (const path of paths) {
    if (!path || path.length < 2) continue;

    // Create line segments between consecutive points in the path
    for (let i = 0; i < path.length - 1; i++) {
      const p1 = getV(path[i]);
      const p2 = getV(path[i + 1]);

      // Add both points to create a line segment
      linePositions.push(...p1, ...p2);
    }
  }

  // Update existing line geometry
  const lineGeom = curveObjects.lines.geometry;

  if (linePositions.length === 0) {
    // Clear lines if no valid lines
    lineGeom.dispose();
    curveObjects.lines.geometry = new THREE.BufferGeometry();
    console.log("updateBasicCurveLines: No line positions - geometry cleared");
  } else {
    // Update line geometry with new positions (basic BufferGeometry)
    const positionArray = new Float32Array(linePositions);
    lineGeom.setAttribute("position", new THREE.BufferAttribute(positionArray, 3));
    lineGeom.attributes.position.needsUpdate = true;
    console.log(
      `updateBasicCurveLines: Updated with ${linePositions.length / 6} line segments (${paths.length} paths)`,
    );
  }

  return curveObjects;
}

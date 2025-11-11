import * as THREE from "three";
import { mesh_object } from '../webAssembly/dist/native.js'
import {LineMaterial} from "three/examples/jsm/lines/LineMaterial";
import {LineSegments2} from "three/examples/jsm/lines/LineSegments2";
import {LineSegmentsGeometry} from "three/examples/jsm/lines/LineSegmentsGeometry";

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


export function createMesh(){
    const material = new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide, flatShading: true });
    const geometry = new THREE.BufferGeometry();
    const mesh = new THREE.Mesh(geometry, material);
    mesh.matrixAutoUpdate = false;
    return mesh;
}

export function getMeshFromWasm(wO: mesh_object, mesh: THREE.Mesh, pointsOnly?: boolean) {
    const pU = wO.polydata_updated;
    if(pU) {
        const geometry = mesh.geometry;
        geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.get_points()), 3));
        geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.get_polys()), 1));
    }
    getMatrixFromWasm(wO, mesh)
}

function getMatrixFromWasm(wO: mesh_object, dstGeometry: THREE.Mesh | THREE.Line) {
    const mU = wO.matrix_updated;
    if(mU) {
        const matrix = new Float32Array(wO.matrix);
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
        console.warn('points length must be multiple of 3');
        return { points: new Float32Array(), paths: [] };
    }
    if (offBuf.length < 2) {
        console.warn('offsets must have at least [0, ids.length]');
        return { points: new Float32Array(), paths: [] };
    }
    if (offBuf[0] !== 0 || offBuf[offBuf.length - 1] !== idBuf.length){
        console.warn('offsets must start at 0 and end at ids.length');
        return { points: new Float32Array(), paths: [] };
    }

    const nPaths = offBuf.length - 1;
    const paths = new Array(nPaths);

    for (let p = 0; p < nPaths; p++) {
        const start = offBuf[p];
        const end   = offBuf[p + 1];
        const len   = Math.max(0, end - start);

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
    const lineMat = new LineMaterial( {
        color: tubeColor,
        linewidth: lineWidth, // in worldUnit
        worldUnits: true,
        vertexColors: false, // Disable vertex colors for now to avoid conflicts
        alphaToCoverage: true, // Disable alpha to coverage for debugging
        transparent: false, // Ensure lines are opaque
        opacity: 1.0,
        depthTest: !alwaysOnTop,
        depthWrite: !alwaysOnTop,

        // --- depth bias to pull the line toward the camera ---
        polygonOffset: true,
        polygonOffsetFactor: -10,
        polygonOffsetUnits:  -2

    } );
    const linesObj = new LineSegments2(lineGeom, lineMat);
    linesObj.name = 'curve_lines';
    linesObj.visible = true;
    linesObj.renderOrder = renderOrder;

    return { lines: linesObj };
}

/**
 * Fast smoothing + line update for many paths each frame.
 * - Smoothing: Chaikin (linear-time), 1–3 iterations recommended.
 * - Reuses typed arrays & geometry attributes to minimize GC.
 */
export function curvesToCurveLinesFast(
    curves: { points: Float32Array; paths: number[][] },
    curveObjects: { lines: THREE.LineSegments },
    opts?: {
        chaikinIterations?: number;       // 0..3; default 2
        closed?: boolean;                 // default false (open path)
    }
) {
    const t0 = performance.now();

    const { points: src, paths } = curves;
    if (!src || !paths) throw new Error('curves must have {points, paths}');

    const iterations = Math.max(0, Math.min(3, opts?.chaikinIterations ?? 2));
    const closed     = !!opts?.closed;

    // ---------------------------------------------------------------------------
    // 2) Workspace (reused between frames) lives on the lines object
    // ---------------------------------------------------------------------------
    type Work = {
        start: Float32Array; end: Float32Array;     // segment ends (xyz xyz …)
        ax: Float32Array; ay: Float32Array; az: Float32Array; // stage A coords
        bx: Float32Array; by: Float32Array; bz: Float32Array; // stage B coords
        segments: number;                            // how many segments written
    };
    const userData = curveObjects.lines.userData as any;
    const work: Work = userData.__work || (userData.__work = {
        start: new Float32Array(0),
        end:   new Float32Array(0),
        ax:    new Float32Array(0),
        ay:    new Float32Array(0),
        az:    new Float32Array(0),
        bx:    new Float32Array(0),
        by:    new Float32Array(0),
        bz:    new Float32Array(0),
        segments: 0
    });

    // helpers to grow reusable buffers
    const ensureLen = (arr: Float32Array, needed: number) =>
        (arr.length >= needed) ? arr : new Float32Array(nextPow2(needed));

    const ensureWorkForPoints = (neededPoints: number) => {
        work.ax = ensureLen(work.ax, neededPoints);
        work.ay = ensureLen(work.ay, neededPoints);
        work.az = ensureLen(work.az, neededPoints);
        work.bx = ensureLen(work.bx, neededPoints * 2); // worst case next iter
        work.by = ensureLen(work.by, neededPoints * 2);
        work.bz = ensureLen(work.bz, neededPoints * 2);
    };

    const ensureStartEndCapacity = (neededSegments: number) => {
        work.start = ensureLen(work.start, neededSegments * 3);
        work.end   = ensureLen(work.end,   neededSegments * 3);
    };

    const nextPow2 = (n: number) => 1 << Math.ceil(Math.log2(Math.max(1, n)));

    // ---------------------------------------------------------------------------
    // 3) Build smoothed segments (Chaikin) directly into start/end arrays
    // ---------------------------------------------------------------------------
    work.segments = 0;

    for (let p = 0; p < paths.length; p++) {
        const idx = paths[p];
        if (!idx || idx.length < 2) continue;

        // Copy current path coordinates into ax/ay/az (stage A) with index validation
        ensureWorkForPoints(idx.length);
        let n = 0; // number of valid points we copied for this path

        const vertCount = src.length / 3;
        for (let i = 0; i < idx.length; i++) {
            const vi = idx[i];
            if (vi == null || vi < 0 || vi >= vertCount) continue; // skip invalid
            const j = 3 * vi;
            work.ax[n] = src[j];
            work.ay[n] = src[j + 1];
            work.az[n] = src[j + 2];
            n++;
        }
        if (n < 2) continue; // nothing to draw for this path

        // Apply Chaikin smoothing 'iterations' times
        // Open and Closed variants
        for (let it = 0; it < iterations; it++) {
            if (!closed) {
                // open curve: preserve endpoints
                // new length: (n - 1) * 2 + 1
                let k = 0;
                work.bx[k] = work.ax[0]; work.by[k] = work.ay[0]; work.bz[k] = work.az[0]; k++;
                for (let i = 0; i < n - 1; i++) {
                    const x0 = work.ax[i],   y0 = work.ay[i],   z0 = work.az[i];
                    const x1 = work.ax[i+1], y1 = work.ay[i+1], z1 = work.az[i+1];
                    // Q and R points (0.25 / 0.75)
                    work.bx[k]   = 0.75 * x0 + 0.25 * x1;
                    work.by[k]   = 0.75 * y0 + 0.25 * y1;
                    work.bz[k++] = 0.75 * z0 + 0.25 * z1;

                    work.bx[k]   = 0.25 * x0 + 0.75 * x1;
                    work.by[k]   = 0.25 * y0 + 0.75 * y1;
                    work.bz[k++] = 0.25 * z0 + 0.75 * z1;
                }
                work.bx[k] = work.ax[n-1]; work.by[k] = work.ay[n-1]; work.bz[k] = work.az[n-1]; k++;
                // swap B->A
                [work.ax, work.ay, work.az, work.bx, work.by, work.bz] =
                    [work.bx, work.by, work.bz, work.ax, work.ay, work.az];
                n = k;
            } else {
                // closed curve: wrap around; new length: n * 2
                let k = 0;
                for (let i = 0; i < n; i++) {
                    const ni = (i + 1) % n;
                    const x0 = work.ax[i],  y0 = work.ay[i],  z0 = work.az[i];
                    const x1 = work.ax[ni], y1 = work.ay[ni], z1 = work.az[ni];

                    work.bx[k]   = 0.75 * x0 + 0.25 * x1;
                    work.by[k]   = 0.75 * y0 + 0.25 * y1;
                    work.bz[k++] = 0.75 * z0 + 0.25 * z1;

                    work.bx[k]   = 0.25 * x0 + 0.75 * x1;
                    work.by[k]   = 0.25 * y0 + 0.75 * y1;
                    work.bz[k++] = 0.25 * z0 + 0.75 * z1;
                }
                [work.ax, work.ay, work.az, work.bx, work.by, work.bz] =
                    [work.bx, work.by, work.bz, work.ax, work.ay, work.az];
                n = k;
            }
        }

        // Number of segments for this path
        const segs = closed ? n : (n - 1);
        ensureStartEndCapacity(work.segments + segs);

        // Write segments into start/end buffers
        let s = work.segments * 3;  // float index
        if (!closed) {
            for (let i = 0; i < n - 1; i++) {
                const x0 = work.ax[i],   y0 = work.ay[i],   z0 = work.az[i];
                const x1 = work.ax[i+1], y1 = work.ay[i+1], z1 = work.az[i+1];
                work.start[s]   = x0; work.start[s+1] = y0; work.start[s+2] = z0;
                work.end[s]     = x1; work.end[s+1]   = y1; work.end[s+2]   = z1;
                s += 3;
            }
        } else {
            for (let i = 0; i < n; i++) {
                const ni = (i + 1) % n;
                const x0 = work.ax[i],  y0 = work.ay[i],  z0 = work.az[i];
                const x1 = work.ax[ni], y1 = work.ay[ni], z1 = work.az[ni];
                work.start[s]   = x0; work.start[s+1] = y0; work.start[s+2] = z0;
                work.end[s]     = x1; work.end[s+1]   = y1; work.end[s+2]   = z1;
                s += 3;
            }
        }
        work.segments += segs;
    }

    // --- 4) Upload to the existing LineSegmentsGeometry (no recreation) ---
    const lineGeom = curveObjects.lines.geometry as LineSegmentsGeometry;
    const needFloats = work.segments * 3;

    let aStart = lineGeom.getAttribute('instanceStart') as THREE.InstancedBufferAttribute | undefined;
    let aEnd = lineGeom.getAttribute('instanceEnd') as THREE.InstancedBufferAttribute | undefined;

// create attributes once
    if (!aStart || !aEnd) {
        aStart = new THREE.InstancedBufferAttribute(work.start, 3).setUsage(THREE.DynamicDrawUsage);
        aEnd = new THREE.InstancedBufferAttribute(work.end, 3).setUsage(THREE.DynamicDrawUsage);
        lineGeom.setAttribute('instanceStart', aStart);
        lineGeom.setAttribute('instanceEnd', aEnd);
    }

    // ensure attribute capacity >= needFloats
    if (aStart.array.length < needFloats || aEnd.array.length < needFloats) {
        // grow our reusable work buffers first
        const newCap = nextPow2(needFloats);
        if (work.start.length < newCap) work.start = new Float32Array(newCap);
        if (work.end.length < newCap) work.end = new Float32Array(newCap);

        // swap arrays on the attributes without recreating them (if supported)
        if ((aStart as any).setArray) {
            (aStart as any).setArray(work.start);
            (aEnd as any).setArray(work.end);
        } else {
            // fallback for older three.js
            lineGeom.setAttribute('instanceStart',
                new THREE.InstancedBufferAttribute(work.start, 3).setUsage(THREE.DynamicDrawUsage));
            lineGeom.setAttribute('instanceEnd',
                new THREE.InstancedBufferAttribute(work.end, 3).setUsage(THREE.DynamicDrawUsage));
            aStart = lineGeom.getAttribute('instanceStart') as THREE.InstancedBufferAttribute;
            aEnd = lineGeom.getAttribute('instanceEnd') as THREE.InstancedBufferAttribute;
        }
    }

// copy only the used range into the attribute arrays
    (aStart.array as Float32Array).set(work.start.subarray(0, needFloats));
    (aEnd.array as Float32Array).set(work.end.subarray(0, needFloats));
    aStart.needsUpdate = true;
    aEnd.needsUpdate = true;

// draw only the number of segments we populated
    (lineGeom as any).instanceCount = work.segments;

// ---- Compute bounds over the USED range only (avoid NaNs in tail) ----
    const used = needFloats; // floats (x,y,z) * number of verts
    const aS = aStart.array as Float32Array;
    const aE = aEnd.array as Float32Array;

// min/max over start + end
    let minX =  Infinity, minY =  Infinity, minZ =  Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i < used; i += 3) {
        let x = aS[i], y = aS[i+1], z = aS[i+2];
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
            if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
            if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
        }
        x = aE[i]; y = aE[i+1]; z = aE[i+2];
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
            if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
            if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
        }
    }

// If nothing finite (shouldn't happen), fall back to disabling culling for this frame
    if (!Number.isFinite(minX)) {
        curveObjects.lines.frustumCulled = false;
    } else {
        const box = new THREE.Box3(
            new THREE.Vector3(minX, minY, minZ),
            new THREE.Vector3(maxX, maxY, maxZ)
        );
        lineGeom.boundingBox = box;

        // sphere: center = box center, radius = max distance of used verts to center
        const center = box.getCenter(new THREE.Vector3());
        let r2 = 0;
        for (let i = 0; i < used; i += 3) {
            let dx = aS[i] - center.x, dy = aS[i+1] - center.y, dz = aS[i+2] - center.z;
            if (Number.isFinite(dx) && Number.isFinite(dy) && Number.isFinite(dz)) {
                const d2 = dx*dx + dy*dy + dz*dz; if (d2 > r2) r2 = d2;
            }
            dx = aE[i] - center.x; dy = aE[i+1] - center.y; dz = aE[i+2] - center.z;
            if (Number.isFinite(dx) && Number.isFinite(dy) && Number.isFinite(dz)) {
                const d2 = dx*dx + dy*dy + dz*dz; if (d2 > r2) r2 = d2;
            }
        }
        lineGeom.boundingSphere = new THREE.Sphere(center, Math.sqrt(r2));
        curveObjects.lines.frustumCulled = true; // back on if you disabled it
    }

    // ---------------------------------------------------------------------------
    // done
    // ---------------------------------------------------------------------------
    console.log(`curvesToCurveLines (fast): ${performance.now() - t0 | 0} ms, segments=${work.segments}`, lineGeom);
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
    opts?: { samplesPerSegment?: number; tension?: number; closed?: boolean }
) {
    const t0 = performance.now();

    const { points: srcPoints, paths } = curves;
    if (!srcPoints || !paths) throw new Error('curves must have {points, paths}');

    const samplesPerSegment = opts?.samplesPerSegment ?? 6; // ↑ for smoother
    const tension = opts?.tension ?? 0.5;                   // centripetal feel
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
            /*closed=*/closedDefault,
            /*type=*/'centripetal',
            /*tension=*/tension
        );

        // resample: original count + (N * (#segments))
        const samples = Math.max(2, (path.length - 1) * samplesPerSegment + 1);
        const smooth: THREE.Vector3[] = curve.getPoints(samples);

        // emit segments between successive resampled points
        for (let i = 0; i < smooth.length - 1; i++) {
            const a = smooth[i], b = smooth[i + 1];
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

    console.log("Time to build line segments: " + (performance.now() - t0) + "ms");
    return curveObjects;
}
/**
 * Alternative implementation using basic THREE.LineSegments for compatibility.
 * Use this if LineSegments2 is not working properly.
 */
export function createBasicCurveLineObjects(opts: curvesToCurvePolyOpts = {}): any {
    const {
        tubeColor = 0xff00ff,
        lineWidth = 2.0
    } = opts;


    // Create basic line segments object
    const lineGeom = new THREE.BufferGeometry();
    const lineMat = new THREE.LineBasicMaterial({
        color: tubeColor,
        linewidth: lineWidth, // Note: linewidth only works on Windows
    });
    const linesObj = new THREE.LineSegments(lineGeom, lineMat);
    linesObj.name = 'curve_lines';
    linesObj.visible = true;

    return { lines: linesObj };
}

/**
 * Updates basic LineSegments geometry (for use with createBasicCurveLineObjects)
 */
export function updateBasicCurveLines(curves: CurveObj, curveObjects: any): any {
    const { points: srcPoints, paths } = curves;
    if (!srcPoints || !paths) throw new Error('curves must have {points, paths}');

    // Build line segments for each polyline path
    const linePositions = [];

    const getV = (i: number) => [
        srcPoints[3 * i],
        srcPoints[3 * i + 1],
        srcPoints[3 * i + 2]
    ];

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
        console.log('updateBasicCurveLines: No line positions - geometry cleared');
    } else {
        // Update line geometry with new positions (basic BufferGeometry)
        const positionArray = new Float32Array(linePositions);
        lineGeom.setAttribute('position', new THREE.BufferAttribute(positionArray, 3));
        lineGeom.attributes.position.needsUpdate = true;
        console.log(`updateBasicCurveLines: Updated with ${linePositions.length/6} line segments (${paths.length} paths)`);
    }

    return curveObjects;
}

import * as THREE from "three";
import { MeshObject } from '../webAssembly/dist/native.js'

export function createMesh(){
    const material = new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide, flatShading: true });
    const geometry = new THREE.BufferGeometry();
    const mesh = new THREE.Mesh(geometry, material);
    mesh.matrixAutoUpdate = false;
    return mesh;
}

export function createLine(){
    const material = new THREE.MeshLambertMaterial({ color: 0x22ccff, side: THREE.DoubleSide, flatShading: true });
    const geometry = new THREE.TubeGeometry();
    const mesh = new THREE.Mesh(geometry, material);
    mesh.matrixAutoUpdate = false;
    return mesh;
}
export function createPoints(){
    const pointPixelSize = 1.0;
    const pointColor = 0xff00ff;
    const pointGeom = new THREE.BufferGeometry();
    const pointMat = new THREE.PointsMaterial({
        color: pointColor,
        size: pointPixelSize,
        sizeAttenuation: true,
    });
    const pointsObj = new THREE.Points(pointGeom, pointMat);
    pointsObj.name = 'curve_points';
    return pointsObj;
}

export function getMeshFromWasm(wO: MeshObject, mesh: THREE.Mesh, pointsOnly?: boolean) {
    const pU = wO.polydataUpdated;
    if(pU) {
        const geometry = mesh.geometry;
        geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.GetPoints()), 3));
        geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.GetPolys()), 1));
    }
    getMatrixFromWasm(wO, mesh)
}
export function getLineFromWasm(wO: MeshObject, mesh: THREE.Points) {
    const pU = wO.polydataUpdated;
    if(pU) {
        const geometry = mesh.geometry;
        geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.GetCurvePoints()), 3));
        // geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.GetCurveIds()), 1));
    }
}

function getMatrixFromWasm(wO: MeshObject, dstGeometry: THREE.Mesh | THREE.Line) {
    const mU = wO.matrixUpdated;
    if(mU) {
        const matrix = new Float32Array(wO.matrix);
        const threeMatrix = new THREE.Matrix4();
        threeMatrix.fromArray(matrix);
        threeMatrix.transpose();
        dstGeometry.matrix = threeMatrix;
    }
}

export interface CurveObj {
    points: Float32Array;
    paths: number[][];
}

export interface CurvePolyObjects {
    curve_poly: THREE.Group;
    tubes: THREE.Mesh;
    points: THREE.Points;
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



import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

/**
 * Creates reusable curve poly objects that can be updated instead of recreated.
 * This is more efficient for animated or frequently changing curves.
 *
 * @param {curvesToCurvePolyOpts} [opts] Configuration options
 * @returns {CurvePolyObjects} Reusable objects for curve visualization
 */
export function createCurvePolyObjects(opts: curvesToCurvePolyOpts = {}): CurvePolyObjects {
    const {
        pointPixelSize = 1.0,
        tubeColor = 0xff00ff,
        pointColor = 0xff00ff
    } = opts;

    // Create point cloud object
    const pointGeom = new THREE.BufferGeometry();
    const pointMat = new THREE.PointsMaterial({
        color: pointColor,
        size: pointPixelSize,
        sizeAttenuation: true,
    });
    const pointsObj = new THREE.Points(pointGeom, pointMat);
    pointsObj.name = 'curve_points';

    // Create tubes mesh object
    const tubeGeom = new THREE.BufferGeometry();
    const tubeMat = new THREE.MeshStandardMaterial({
        color: tubeColor,
        metalness: 0.0,
        roughness: 1.0,
        side: THREE.FrontSide,
    });
    const tubesMesh = new THREE.Mesh(tubeGeom, tubeMat);
    tubesMesh.name = 'curve_tubes';

    // Create container group
    const curve_poly = new THREE.Group();
    curve_poly.name = 'curve_poly';
    curve_poly.add(tubesMesh);
    // curve_poly.add(pointsObj);

    return { curve_poly, tubes: tubesMesh, points: pointsObj };
}


export interface curvesToCurvePolyOpts {
    radius?: number;
    radialSegments?: number;
    tubularSegPerEdge?: number;
    pointPixelSize?: number;
    tubeColor?: number;
    pointColor?: number;
    pointsOnly?: boolean;
}

/**
 * Three.js equivalent of:
 *   - building vtkPolyData from {points, paths}
 *   - running vtkTubeFilter with SetRadius(0.05)
 *
 * Updates existing CurvePolyObjects with new curve data instead of creating new objects.
 * This is more efficient for animated or frequently changing curves.
 *
 * @param {Object} curves
 * @param {Array|Float32Array} curves.points  // [[x,y,z], ...] or Float32Array length 3N
 * @param {Array<Array<number>>} curves.paths // [[i0,i1,i2,...], ...]
 * @param {CurvePolyObjects} curveObjects Pre-created objects to update
 * @param {Object} [opts]
 * @param {number} [opts.radius=0.05]         // tube radius, world units (VTK: SetRadius)
 * @param {number} [opts.radialSegments=12]   // sides around the tube (VTK: NumberOfSides)
 * @param {number} [opts.tubularSegPerEdge=8] // longitudinal segments per path edge
 * @returns {CurvePolyObjects} The updated objects
 */
export function curvesToCurvePoly(curves: CurveObj, curveObjects: CurvePolyObjects, opts: curvesToCurvePolyOpts = {}): CurvePolyObjects {
    const { radius = 0.5,
        radialSegments = 12,
        tubularSegPerEdge = 8,
        pointsOnly = false,
    } = opts;

    const { points: srcPoints, paths } = curves;
    if (!srcPoints || !paths) throw new Error('curves must have {points, paths}');

    // Update existing point cloud geometry
    const pointGeom = curveObjects.points.geometry;
    pointGeom.setAttribute('position', new THREE.BufferAttribute(srcPoints, 3));
    pointGeom.attributes.position.needsUpdate = true;

    if(pointsOnly) return curveObjects;

    // --- Build tubes for each polyline path (vtkTubeFilter analogue)
    const tubeGeoms = [];
    const getV = (i: number) => new THREE.Vector3(
        srcPoints[3 * i],
        srcPoints[3 * i + 1],
        srcPoints[3 * i + 2]
    );

    for (const path of paths) {
        if (!path || path.length < 2) continue;

        // Create a polyline curve using THREE.CatmullRomCurve3 for smooth tubes
        const pathPoints = path.map(getV);

        // Skip paths that don't have enough points or have degenerate geometry
        if (pathPoints.length < 2) continue;

        // Create curve from path points
        const curve = new THREE.CatmullRomCurve3(pathPoints, false, 'centripetal');

        const tubularSegments = Math.max(1, path.length * tubularSegPerEdge);
        const tubeGeom = new THREE.TubeGeometry(
            curve,            // curve
            tubularSegments,  // segments along the length
            radius,           // radius
            radialSegments,   // segments around
            /*closed*/ false  // keep open unless your path is cyclic
        );
        tubeGeoms.push(tubeGeom);
    }

    // Update existing tubes geometry
    const tubesGeometry = curveObjects.tubes.geometry;

    if (tubeGeoms.length === 0) {
        // Clear tubes if no valid tubes
        tubesGeometry.dispose();
        curveObjects.tubes.geometry = new THREE.BufferGeometry();
    } else {
        // Merge all tubes into one geometry and update existing mesh
        const mergedTubeGeom = mergeGeometries(tubeGeoms, /*useGroups*/ false);

        // Dispose old geometry and assign new one
        tubesGeometry.dispose();
        curveObjects.tubes.geometry = mergedTubeGeom;
    }

    return curveObjects;
}
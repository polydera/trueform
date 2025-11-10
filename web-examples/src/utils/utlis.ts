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
        if(pU) {
            const geometry = mesh.geometry;
            geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.GetPoints()), 3));
            geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.GetPolys()), 1));
        }
    }
    getMatrixFromWasm(wO, mesh)
}
export function getLineFromWasm(wO: MeshObject, mesh: THREE.Points) {
    const pU = wO.polydataUpdated;
    if(pU) {
        if(pU) {
            const geometry = mesh.geometry;
            geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.GetCurvePoints()), 3));
            // geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.GetCurveIds()), 1));
        }
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
    if (points.length % 3 !== 0) throw new Error('points length must be multiple of 3');
    if (offBuf.length < 2) throw new Error('offsets must have at least [0, ids.length]');
    if (offBuf[0] !== 0 || offBuf[offBuf.length - 1] !== idBuf.length)
        throw new Error('offsets must start at 0 and end at ids.length');

    const nPaths = offBuf.length - 1;
    const paths = new Array(nPaths);

    for (let p = 0; p < nPaths; p++) {
        const start = offBuf[p];
        const end   = offBuf[p + 1];
        const len   = Math.max(0, end - start);

        const path = new Array(len);
        for (let i = 0; i < len; i++) {
            const idx = idBuf[start + i];
            path[i] = idx;
        }
        paths[p] = path;
    }
    return { points, paths };
}



import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';


export interface curvesToCurvePolyOpts {
    radius?: number;
    radialSegments?: number;
    tubularSegPerEdge?: number;
    pointPixelSize?: number;
    tubeColor?: number;
    pointColor?: number;
}
/**
 * Three.js equivalent of:
 *   - building vtkPolyData from {points, paths}
 *   - running vtkTubeFilter with SetRadius(0.05)
 *
 * Returns a Group named 'curve_poly' that contains:
 *   - tubes mesh (merged TubeGeometry)
 *   - points cloud (THREE.Points)
 *
 * @param {Object} curves
 * @param {Array|Float32Array} curves.points  // [[x,y,z], ...] or Float32Array length 3N
 * @param {Array<Array<number>>} curves.paths // [[i0,i1,i2,...], ...]
 * @param {Object} [opts]
 * @param {number} [opts.radius=0.05]         // tube radius, world units (VTK: SetRadius)
 * @param {number} [opts.radialSegments=12]   // sides around the tube (VTK: NumberOfSides)
 * @param {number} [opts.tubularSegPerEdge=8] // longitudinal segments per path edge
 * @param {number} [opts.pointPixelSize=3.0]  // Points size in pixels
 * @param {number} [opts.tubeColor=0x4287f5]
 * @param {number} [opts.pointColor=0xffffff]
 * @returns {{curve_poly:THREE.Group, tubes:THREE.Mesh, points:THREE.Points}}
 */
export function curvesToCurvePoly(curves: CurveObj, opts: curvesToCurvePolyOpts = {}) {
    const { radius = 0.05,
        radialSegments = 20,
        tubularSegPerEdge = 50,
        pointPixelSize = 1.0,
        tubeColor = 0xff00ff,
        pointColor = 0xff00ff } = opts;

    const { points: srcPoints, paths } = curves;
    if (!srcPoints || !paths) throw new Error('curves must have {points, paths}');

    // --- Build THREE.BufferGeometry for the point cloud (vtkPoints analogue)
    const pointGeom = new THREE.BufferGeometry();
    pointGeom.setAttribute('position', new THREE.BufferAttribute(srcPoints, 3));

    const pointMat = new THREE.PointsMaterial({
        color: pointColor,
        size: pointPixelSize,
        sizeAttenuation: true,
    });
    const pointsObj = new THREE.Points(pointGeom, pointMat);
    pointsObj.name = 'curve_points';
    //
    // // --- Build tubes for each polyline path (vtkTubeFilter analogue)
    // const tubeGeoms = [];
    // const getV = (i: number) => new THREE.Vector3(
    //     srcPoints[3 * i + 0],
    //     srcPoints[3 * i + 1],
    //     srcPoints[3 * i + 2]
    // );
    //
    // for (const path of paths) {
    //     if (!path || path.length < 2) continue;
    //
    //     // Build a polyline curve out of straight segments to preserve original vertices.
    //     const cp = new THREE.CurvePath();
    //     for (let i = 0; i < path.length - 1; i++) {
    //         const a = getV(path[i]);
    //         const b = getV(path[i + 1]);
    //         // Skip degenerate edges
    //         if (a.distanceToSquared(b) === 0) continue;
    //         cp.add(new THREE.LineCurve3(a, b));
    //     }
    //
    //     // If everything degenerated, skip.
    //     if (cp.curves.length === 0) continue;
    //
    //     const tubularSegments = Math.max(1, cp.curves.length * tubularSegPerEdge);
    //     const tubeGeom = new THREE.TubeGeometry(
    //         cp,               // curve
    //         tubularSegments,  // segments along the length
    //         radius,           // radius (VTK: SetRadius)
    //         radialSegments,   // segments around
    //         /*closed*/ false  // keep open unless your path is cyclic
    //     );
    //     tubeGeoms.push(tubeGeom);
    // }
    //
    // if (tubeGeoms.length === 0) {
    //     // No valid tubes; still return points for parity with VTK pipeline.
    //     const emptyMesh = new THREE.Mesh(new THREE.BufferGeometry());
    //     emptyMesh.name = 'curve_tubes';
    //     const group = new THREE.Group();
    //     group.name = 'curve_poly';
    //     group.add(pointsObj);
    //     group.add(emptyMesh);
    //     return { curve_poly: group, tubes: emptyMesh, points: pointsObj };
    // }
    //
    // // Merge all tubes into one geometry to minimize draw calls (like one vtkPolyData)
    // const mergedTubeGeom = mergeGeometries(tubeGeoms, /*useGroups*/ false);
    // const tubeMat = new THREE.MeshStandardMaterial({
    //     color: tubeColor,
    //     metalness: 0.0,
    //     roughness: 1.0,
    //     side: THREE.FrontSide,
    // });
    const tubesMesh = new THREE.Mesh();
    // const tubesMesh = new THREE.Mesh(mergedTubeGeom, tubeMat);
    // tubesMesh.name = 'curve_tubes';
    //
    // // Final container analogous to your `curve_poly`
    const curve_poly = new THREE.Group();
    // curve_poly.name = 'curve_poly';
    // curve_poly.add(tubesMesh);
    // curve_poly.add(pointsObj);

    return { curve_poly, tubes: tubesMesh, points: pointsObj };
}
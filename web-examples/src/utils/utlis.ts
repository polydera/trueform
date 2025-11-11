import * as THREE from "three";
import { mesh_object } from '../webAssembly/dist/native.js'

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

export function getMeshFromWasm(wO: mesh_object, mesh: THREE.Mesh, pointsOnly?: boolean) {
    const pU = wO.polydata_updated;
    if(pU) {
        const geometry = mesh.geometry;
        geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.GetPoints()), 3));
        geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.GetPolys()), 1));
    }
    getMatrixFromWasm(wO, mesh)
}
export function getLineFromWasm(wO: mesh_object, mesh: THREE.Points) {
    const pU = wO.polydata_updated;
    if(pU) {
        const geometry = mesh.geometry;
        geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.GetCurvePoints()), 3));
        // geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.GetCurveIds()), 1));
    }
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

export interface CurveObj {
    points: Float32Array;
    paths: number[][];
}

export interface CurvePolyObjects {
    curve_poly: THREE.Group;
    tubes: THREE.Mesh;
    points: THREE.Points;
}

export interface CurveLineObjects {
    curve_lines: THREE.Group;
    lines: LineSegments2;
    // points: THREE.Points;
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
import {LineMaterial} from "three/examples/jsm/lines/LineMaterial";
import {LineSegments2} from "three/examples/jsm/lines/LineSegments2";
import {LineSegmentsGeometry} from "three/examples/jsm/lines/LineSegmentsGeometry";

/**
 * Creates reusable curve line objects that can be updated instead of recreated.
 * This is much more efficient than TubeGeometry for performance-critical applications.
 *
 * @param {curvesToCurvePolyOpts} [opts] Configuration options
 * @returns {CurveLineObjects} Reusable objects for line-based curve visualization
 */
export function createCurveLineObjects(opts: curvesToCurvePolyOpts = {}): CurveLineObjects {
    const {
        pointPixelSize = 1.0,
        tubeColor = 0xff00ff,
        pointColor = 0xff00ff,
        lineWidth = 2.0
    } = opts;

    // // Create point cloud object
    // const pointGeom = new THREE.BufferGeometry();
    // const pointMat = new THREE.PointsMaterial({
    //     color: pointColor,
    //     size: pointPixelSize,
    //     sizeAttenuation: true,
    // });
    // const pointsObj = new THREE.Points(pointGeom, pointMat);
    // pointsObj.name = 'curve_points';

    // Create line segments object
    const lineGeom = new LineSegmentsGeometry();
    const lineMat = new LineMaterial( {
        color: tubeColor,
        linewidth: lineWidth, // in pixels when worldUnits is false
        worldUnits: false, // Use pixel units for more predictable behavior
        vertexColors: false, // Disable vertex colors for now to avoid conflicts
        alphaToCoverage: false, // Disable alpha to coverage for debugging
        transparent: false, // Ensure lines are opaque
        opacity: 1.0,
        depthTest: true,
        depthWrite: true,
        resolution: new THREE.Vector2(window.innerWidth, window.innerHeight), // Use actual screen resolution
    } );
    const linesObj = new LineSegments2(lineGeom, lineMat);
    linesObj.name = 'curve_lines';
    linesObj.visible = true;

    // Create container group
    const curve_lines = new THREE.Group();
    curve_lines.name = 'curve_lines';
    curve_lines.add(linesObj);

    return { curve_lines, lines: linesObj };
}

/**
 * Updates existing CurveLineObjects with new curve data using LineSegments instead of TubeGeometry.
 * This is much faster than tube-based rendering and suitable for performance-critical applications.
 *
 * @param {Object} curves
 * @param {Array|Float32Array} curves.points  // [[x,y,z], ...] or Float32Array length 3N
 * @param {Array<Array<number>>} curves.paths // [[i0,i1,i2,...], ...]
 * @param {CurveLineObjects} curveObjects Pre-created objects to update
 * @param {Object} [opts]
 * @returns {CurveLineObjects} The updated objects
 */
export function curvesToCurveLines(curves: CurveObj, curveObjects: CurveLineObjects, opts: curvesToCurvePolyOpts = {}): CurveLineObjects {
    const { pointsOnly = false } = opts;

    const { points: srcPoints, paths } = curves;
    if (!srcPoints || !paths) throw new Error('curves must have {points, paths}');

    // // Update existing point cloud geometry
    // const pointGeom = curveObjects.points.geometry;
    // pointGeom.setAttribute('position', new THREE.BufferAttribute(srcPoints, 3));
    // pointGeom.attributes.position.needsUpdate = true;
    //
    // if (pointsOnly) return curveObjects;

    // Build line segments for each polyline path
    const linePositions = [];

    const getV = (i: number) => [
        srcPoints[3 * i],
        srcPoints[3 * i + 1],
        srcPoints[3 * i + 2]
    ];

    for (const path of paths) {
        if (!path || path.length < 2) {
            console.log(`Skipping path with length: ${path ? path.length : 'null'}`);
            continue;
        }

        console.log(`Processing path with ${path.length} points`);

        // Create line segments between consecutive points in the path
        for (let i = 0; i < path.length - 1; i++) {
            const p1 = getV(path[i]);
            const p2 = getV(path[i + 1]);

            // Add both points to create a line segment
            linePositions.push(...p1, ...p2);
        }
    }

    console.log(`Total paths processed: ${paths.length}, total line positions: ${linePositions.length}`);
    if (linePositions.length > 0) {
        console.log(`First few positions: [${linePositions.slice(0, 12).join(', ')}]`);
    }

    // Update existing line geometry
    const lineGeom = curveObjects.lines.geometry as LineSegmentsGeometry;

    if (linePositions.length === 0) {
        // Clear lines if no valid lines
        lineGeom.dispose();
        curveObjects.lines.geometry = new LineSegmentsGeometry();
        console.log('curvesToCurveLines: No line positions - geometry cleared');
    } else {
        // Create a new LineSegmentsGeometry instead of trying to reuse the old one
        const newLineGeom = new LineSegmentsGeometry();

        try {
            // Method 1: Try using fromLineSegments (recommended approach)
            const tempGeometry = new THREE.BufferGeometry();
            const positionArray = new Float32Array(linePositions);
            tempGeometry.setAttribute('position', new THREE.BufferAttribute(positionArray, 3));

            // Create a temporary LineSegments object
            const tempLineSegments = new THREE.LineSegments(tempGeometry);

            // Use fromLineSegments to properly convert
            newLineGeom.fromLineSegments(tempLineSegments);

            // Clean up
            tempGeometry.dispose();

            console.log(`Using fromLineSegments method with ${linePositions.length/6} line segments`);
            console.log(`Temp geometry had ${tempGeometry.attributes.position?.count || 0} position vertices`);
        } catch (error) {
            console.error('fromLineSegments failed:', error);

            // Method 2: Create the geometry manually using setPositions
            try {
                const positionArray = new Float32Array(linePositions);
                newLineGeom.setPositions(positionArray);
                console.log(`Using setPositions method with ${linePositions.length/6} line segments`);
            } catch (error2) {
                console.error('setPositions also failed:', error2);

                // Method 3: Last resort - setFromPoints with pairs
                const pointsVector3 = [];
                for (let i = 0; i < linePositions.length; i += 3) {
                    pointsVector3.push(new THREE.Vector3(
                        linePositions[i],
                        linePositions[i + 1],
                        linePositions[i + 2]
                    ));
                }
                newLineGeom.setFromPoints(pointsVector3);
                console.log(`Using setFromPoints fallback method with ${pointsVector3.length} points`);
            }
        }

        // Validate the created geometry
        if (!newLineGeom.attributes.instanceStart || newLineGeom.attributes.instanceStart.count === 0) {
            console.warn('LineSegmentsGeometry creation failed - trying manual approach');

            // Manual approach: Create instanceStart and instanceEnd attributes directly
            const numSegments = linePositions.length / 6;
            const instanceStart = new Float32Array(numSegments * 3);
            const instanceEnd = new Float32Array(numSegments * 3);

            for (let i = 0; i < numSegments; i++) {
                const startIdx = i * 6;
                // Start point
                instanceStart[i * 3] = linePositions[startIdx];
                instanceStart[i * 3 + 1] = linePositions[startIdx + 1];
                instanceStart[i * 3 + 2] = linePositions[startIdx + 2];
                // End point
                instanceEnd[i * 3] = linePositions[startIdx + 3];
                instanceEnd[i * 3 + 1] = linePositions[startIdx + 4];
                instanceEnd[i * 3 + 2] = linePositions[startIdx + 5];
            }

            newLineGeom.setAttribute('instanceStart', new THREE.InstancedBufferAttribute(instanceStart, 3));
            newLineGeom.setAttribute('instanceEnd', new THREE.InstancedBufferAttribute(instanceEnd, 3));

            // Add dummy position attribute (required by LineSegments2)
            newLineGeom.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 1, 0, 0], 3));

            console.log(`Manual LineSegmentsGeometry created with ${numSegments} segments`);
        }

        // Replace the old geometry with the new one
        lineGeom.dispose();
        curveObjects.lines.geometry = newLineGeom;

        console.log(`curvesToCurveLines: Updated with ${linePositions.length/6} line segments (${paths.length} paths)`);

        // Debug geometry information
        const attributes = newLineGeom.attributes;
        console.log('LineSegmentsGeometry attributes:', Object.keys(attributes));
        console.log(`Expected line segments: ${linePositions.length/6}`);
        console.log(`Input positions length: ${linePositions.length}`);

        if (attributes.position) {
            console.log(`Position attribute count: ${attributes.position.count}`);
        }
        if (attributes.instanceStart) {
            console.log(`InstanceStart count: ${attributes.instanceStart.count}`);
        }
        if (attributes.instanceEnd) {
            console.log(`InstanceEnd count: ${attributes.instanceEnd.count}`);
        }
        if (attributes.instanceColorStart) {
            console.log(`InstanceColorStart count: ${attributes.instanceColorStart.count}`);
        }
        if (attributes.instanceColorEnd) {
            console.log(`InstanceColorEnd count: ${attributes.instanceColorEnd.count}`);
        }

        // Check if geometry is actually valid for rendering
        console.log(`Geometry drawRange: count=${newLineGeom.drawRange.count}, start=${newLineGeom.drawRange.start}`);

        // Verify the LineSegments2 object itself
        console.log(`LineSegments2 visible: ${curveObjects.lines.visible}`);
        console.log(`LineSegments2 frustumCulled: ${curveObjects.lines.frustumCulled}`);
    }

    return curveObjects;
}

/**
 * Updates the resolution of LineMaterial for proper rendering.
 * Call this whenever the renderer size changes.
 *
 * @param curveObjects CurveLineObjects containing the LineMaterial to update
 * @param renderer The WebGL renderer to get the size from
 */
export function updateCurveLineResolution(curveObjects: CurveLineObjects, renderer: THREE.WebGLRenderer) {
    const material = curveObjects.lines.material as LineMaterial;
    const size = renderer.getSize(new THREE.Vector2());

    // Ensure we have valid dimensions
    if (size.x > 0 && size.y > 0) {
        material.resolution.copy(size);
        console.log(`Updated LineMaterial resolution to: ${size.x} x ${size.y}`);
    } else {
        // Fallback to window size
        material.resolution.set(window.innerWidth, window.innerHeight);
        console.log(`Using fallback resolution: ${window.innerWidth} x ${window.innerHeight}`);
    }
}

/**
 * Alternative implementation using basic THREE.LineSegments for compatibility.
 * Use this if LineSegments2 is not working properly.
 */
export function createBasicCurveLineObjects(opts: curvesToCurvePolyOpts = {}): any {
    const {
        pointPixelSize = 1.0,
        tubeColor = 0xff00ff,
        pointColor = 0xff00ff,
        lineWidth = 2.0
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

    // Create basic line segments object
    const lineGeom = new THREE.BufferGeometry();
    const lineMat = new THREE.LineBasicMaterial({
        color: tubeColor,
        linewidth: lineWidth, // Note: linewidth only works on Windows
    });
    const linesObj = new THREE.LineSegments(lineGeom, lineMat);
    linesObj.name = 'curve_lines';
    linesObj.visible = true;

    // Create container group
    const curve_lines = new THREE.Group();
    curve_lines.name = 'curve_lines';
    curve_lines.add(linesObj);

    return { curve_lines, lines: linesObj, points: pointsObj };
}

/**
 * Updates basic LineSegments geometry (for use with createBasicCurveLineObjects)
 */
export function updateBasicCurveLines(curves: CurveObj, curveObjects: any, opts: curvesToCurvePolyOpts = {}): any {
    const { pointsOnly = false } = opts;

    const { points: srcPoints, paths } = curves;
    if (!srcPoints || !paths) throw new Error('curves must have {points, paths}');

    // Update existing point cloud geometry
    const pointGeom = curveObjects.points.geometry;
    pointGeom.setAttribute('position', new THREE.BufferAttribute(srcPoints, 3));
    pointGeom.attributes.position.needsUpdate = true;

    if (pointsOnly) return curveObjects;

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

export interface curvesToCurvePolyOpts {
    radius?: number;
    radialSegments?: number;
    tubularSegPerEdge?: number;
    pointPixelSize?: number;
    tubeColor?: number;
    pointColor?: number;
    pointsOnly?: boolean;
    lineWidth?: number;
}
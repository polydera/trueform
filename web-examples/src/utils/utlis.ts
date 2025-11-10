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

export function getMeshFromWasm(wO: MeshObject, mesh: THREE.Mesh) {
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

export function getLineFromWasm(wO: MeshObject, line: THREE.Line) {
    const pU = wO.polydataUpdated;
    if(pU) {
        if(pU) {
            const geometry = line.geometry;
            geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.GetPoints()), 3));
            geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.GetPolys()), 1));
        }
    }
    getMatrixFromWasm(wO, line)
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
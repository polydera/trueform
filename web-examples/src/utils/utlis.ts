import * as THREE from "three";
import { MeshObject } from '../webAssembly/dist/native.js'

export function createMesh(){
    const material = new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide, flatShading: true });
    const geometry = new THREE.BufferGeometry();
    const mesh = new THREE.Mesh(geometry, material);
    mesh.matrixAutoUpdate = false;
    return mesh;
}

export function getMeshFromWasm(wO: MeshObject, mesh: THREE.Mesh) {
    const pU = wO.polydataUpdated;
    const mU = wO.matrixUpdated;
    if(pU || mU) {
        if(pU) {
            const geometry = mesh.geometry;
            geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wO.GetPoints()), 3));
            geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wO.GetPolys()), 1));
        }
        if(mU) {
            const matrix = new Float32Array(wO.matrix);
            const threeMatrix = new THREE.Matrix4();
            threeMatrix.fromArray(matrix);
            threeMatrix.transpose();
            mesh.matrix = threeMatrix;
        }
    }
}
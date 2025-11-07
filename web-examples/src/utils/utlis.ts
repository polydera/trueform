import * as THREE from "three";

export function getMeshFromWasm(wasmObject: {outputPoints: ArrayBuffer, outputPolys: ArrayBuffer}, geometry: THREE.BufferGeometry) {
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(wasmObject.outputPoints), 3));
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(wasmObject.outputPolys), 3));
    geometry.computeVertexNormals();
}
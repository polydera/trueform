/**
 * Raycast rendering example
 *
 * Renders a mesh to a BMP image using batch ray casting and Lambertian shading.
 * All shading is computed with NDArray operations.
 *
 * Usage:
 *     node raycast_render.mjs [mesh.stl]
 *
 * Default mesh: ../../benchmarks/data/dragon-250k.stl
 * Output: output.bmp
 */

import * as tf from "@polydera/trueform";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// =========================================================================
// Load mesh
// =========================================================================

const defaultMesh = resolve(__dirname, "../../benchmarks/data/dragon-250k.stl");
const meshPath = process.argv[2] || defaultMesh;

console.log(`Loading mesh: ${meshPath}`);
const data = readFileSync(resolve(meshPath));
const mesh = meshPath.endsWith(".obj") ? tf.readObj(data) : tf.readStl(data);

console.log(`${mesh.numberOfFaces} faces, ${mesh.numberOfPoints} points`);
mesh.buildTree();

// =========================================================================
// Compute bounding box and camera setup
// =========================================================================

const ptsMin = tf.min(mesh.points, 0);
const ptsMax = tf.max(mesh.points, 0);
const minData = ptsMin.data;
const maxData = ptsMax.data;

const center = [
  (minData[0] + maxData[0]) / 2,
  (minData[1] + maxData[1]) / 2,
  (minData[2] + maxData[2]) / 2,
];
const extent = [
  maxData[0] - minData[0],
  maxData[1] - minData[1],
  maxData[2] - minData[2],
];
const diagonal = Math.sqrt(extent[0] ** 2 + extent[1] ** 2 + extent[2] ** 2);
const halfSpan = Math.max(extent[0], extent[1]) / 2 * 1.1; // 10% margin around XY AABB

ptsMin.delete();
ptsMax.delete();

console.log(`Center: [${center.map((v) => v.toFixed(3)).join(", ")}]`);
console.log(`Diagonal: ${diagonal.toFixed(3)}`);

mesh.normals;
mesh.buildTree();

// =========================================================================
// Build orthographic ray grid
// =========================================================================

const n = 1024;
console.log(`\nRendering ${n}x${n} image...`);

// Camera looks along -Z
const u = tf.linspace(-halfSpan, halfSpan, n);

// Meshgrid: gx varies across columns, gy varies across rows
const gx = tf.tile(u, [n]);
const uc = u.clone();
uc.shape = [n, 1];
const gy = tf.tile(uc, [1, n]).flatten();

// Ray origins: grid on XY plane, offset far in +Z
const originZ = center[2] + diagonal * 2;
const ox = gx.add(center[0]);
const oy = gy.add(center[1]);
const oz = tf.full("float32", [n * n], originZ);
const origins = tf.stack([ox, oy, oz], 1);

// All rays point in -Z direction
const dir1 = tf.ndarray([0, 0, -1], [1, 3]);
const dirs = tf.tile(dir1, [n * n, 1]);
const rays = tf.ray(origins, dirs);

// =========================================================================
// Cast rays
// =========================================================================

const t0 = performance.now();
const { hits, elementIds } = tf.rayCast(rays, mesh);
const elapsed = (performance.now() - t0).toFixed(1);

const nHits = tf.sum(hits);
console.log(`Hits: ${nHits} / ${n * n}`);

// =========================================================================
// Lambertian shading (all NDArray ops)
// =========================================================================

// Face normals indexed by hit element IDs
const normals = mesh.normals;
const hitNormals = normals.take(elementIds, 0); // [n*n, 3]

// Two directional lights + ambient — tf.dot broadcasts [n*n, 3] x [1, 3] -> [n*n]
const light0 = tf.ndarray([0.577, 0.577, 0.577], [1, 3]);
const light1 = tf.ndarray([-0.707, 0.707, 0], [1, 3]);

const d0 = tf.dot(hitNormals, light0).clip(0, 1).mul(0.7);
const d1 = tf.dot(hitNormals, light1).clip(0, 1).mul(0.3);
const shade = d0.add(d1).add(0.15).clip(0, 1);

light0.delete(); light1.delete();
d0.delete(); d1.delete();

// Compose BGR image as NDArray [n*n, 3], rounded to integers
const bCh = tf.where(hits, shade.mul(190), tf.full("float32", [n * n], 52));
const gCh = tf.where(hits, shade.mul(213), tf.full("float32", [n * n], 43));
const rCh = tf.where(hits, shade.mul(0),   tf.full("float32", [n * n], 27));
const bgr = tf.round(tf.stack([bCh, gCh, rCh], 1).clip(0, 255));
const int8 = bgr.as("int8").data; // Int8Array (same bit pattern as uint8)
const pixels = Buffer.from(new Uint8Array(int8.buffer, int8.byteOffset, int8.byteLength));

// =========================================================================
// Write BMP
// =========================================================================

const pixelDataSize = n * n * 3; // no padding needed when n*3 is divisible by 4
const headerSize = 54;

const header = Buffer.alloc(headerSize);

// File header (14 bytes)
header.write("BM", 0);
header.writeUInt32LE(headerSize + pixelDataSize, 2);
header.writeUInt32LE(0, 6);
header.writeUInt32LE(headerSize, 10);

// DIB header (40 bytes)
header.writeUInt32LE(40, 14);
header.writeInt32LE(n, 18);
header.writeInt32LE(n, 22);
header.writeUInt16LE(1, 26);
header.writeUInt16LE(24, 28);
header.writeUInt32LE(0, 30);
header.writeUInt32LE(pixelDataSize, 34);
header.writeInt32LE(2835, 38);
header.writeInt32LE(2835, 42);
header.writeUInt32LE(0, 46);
header.writeUInt32LE(0, 50);

const outPath = resolve(__dirname, "output.bmp");
writeFileSync(outPath, Buffer.concat([header, pixels]));

console.log(`\nRay cast: ${elapsed} ms`);
console.log(`Written to ${outPath}`);

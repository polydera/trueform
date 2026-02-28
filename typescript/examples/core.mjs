/**
 * Core functionality examples
 *
 * Demonstrates NDArray operations, primitive queries, spatial queries,
 * boolean operations, isocontours, and transformations.
 *
 * Usage:
 *     node core.mjs
 */

import * as tf from "@polydera/trueform";

// =========================================================================
// 1. NDArray
// =========================================================================
console.log("=".repeat(60));
console.log("=== NDArray ===");
console.log("=".repeat(60));

// Create arrays
const a = tf.ndarray([1, 2, 3, 4, 5, 6], [2, 3]);
console.log("a shape:", a.shape, "data:", a.data);

const b = tf.linspace(0, 1, 5);
console.log("linspace(0, 1, 5):", b.data);

const c = tf.arange("float32", 0, 10, 2);
console.log("arange(0, 10, 2):", c.data);

const r = tf.random("float32", [2, 3]);
console.log("random [2,3]:", r.data);

// Reductions
const vals = tf.ndarray([1, 2, 3, 4, 5]);
console.log("sum:", tf.sum(vals));
console.log("mean:", tf.mean(vals));
console.log("min:", tf.min(vals), "max:", tf.max(vals));

// Element-wise math
const x = tf.ndarray([1, 4, 9]);
console.log("sqrt:", tf.sqrt(x).data);
console.log("norm:", tf.norm(x));

// Stack
const v1 = tf.ndarray([1, 2, 3]);
const v2 = tf.ndarray([4, 5, 6]);
const stacked = tf.stack([v1, v2], 0);
console.log("stack shape:", stacked.shape, "data:", stacked.data);

// Boolean indexing
const arr = tf.ndarray([10, 20, 30, 40, 50]);
const mask = arr.gt(25);
console.log("mask (>25):", mask.data);
const filtered = arr.booleanIndex(mask);
console.log("filtered:", filtered.data);

// Cleanup
a.delete(); b.delete(); c.delete(); r.delete();
vals.delete(); x.delete(); v1.delete(); v2.delete();
stacked.delete(); arr.delete(); mask.delete(); filtered.delete();

// =========================================================================
// 2. Queries on Primitives
// =========================================================================
console.log("\n" + "=".repeat(60));
console.log("=== Queries on Primitives ===");
console.log("=".repeat(60));

// Create primitives
const triangle = tf.triangle([0, 0, 0], [1, 0, 0], [0, 1, 0]);
const segment = tf.segment([2, 2, 0], [3, 3, 0]);

// Intersection test
const triSegIntersects = tf.intersects(triangle, segment);
console.log("Triangle-segment intersects:", triSegIntersects);

if (!triSegIntersects) {
  const pair = tf.closestPointPair(triangle, segment);
  console.log("Distance:", Math.sqrt(pair.distance2));
  console.log("Point on triangle:", pair.point0.data);
  console.log("Point on segment:", pair.point1.data);
  pair.point0.delete();
  pair.point1.delete();
}

// Ray casting
const ray = tf.ray([0.2, 0.2, -1], [0, 0, 1]);
const hit = tf.rayCast(ray, triangle);
console.log("Ray hit:", hit.hit, "t:", hit.t);

triangle.delete();
segment.delete();
ray.delete();

// =========================================================================
// 3. Spatial Queries on Meshes
// =========================================================================
console.log("\n" + "=".repeat(60));
console.log("=== Spatial Queries on Meshes ===");
console.log("=".repeat(60));

// Create a sphere mesh
const sphere = tf.sphereMesh(1.0, 32, 32);
console.log(`Sphere: ${sphere.numberOfFaces} faces, ${sphere.numberOfPoints} points`);

// Build spatial tree
sphere.buildTree();

// Single nearest neighbor
const query = tf.point(0.5, 0.5, 0.5);
const nn = tf.neighborSearch(sphere, query);
console.log(`Nearest face: ${nn.elementId}, distance: ${Math.sqrt(nn.distance2).toFixed(4)}`);
console.log("Closest point:", nn.point.data);
nn.point.delete();

// k-NN search
const knn = tf.neighborSearch(sphere, query, { k: 3 });
console.log(`k-NN (k=3): ${knn.elementIds.data.length} results`);
console.log("  face IDs:", knn.elementIds.data);
knn.elementIds.delete();
knn.points.delete();
knn.distances.delete();

// Collision detection between meshes
const sphere2 = tf.sphereMesh(1.0, 16, 16);
sphere2.transformation = tf.makeTranslation(0.5, 0, 0);
sphere2.buildTree();

console.log("Spheres intersect:", tf.intersects(sphere, sphere2));

query.delete();
sphere.delete();
sphere2.delete();

// =========================================================================
// 4. Boolean Operations
// =========================================================================
console.log("\n" + "=".repeat(60));
console.log("=== Boolean Operations ===");
console.log("=".repeat(60));

// Create two overlapping boxes
const box0 = tf.boxMesh(1, 1, 1);
const box1 = tf.boxMesh(1, 1, 1);
box1.transformation = tf.makeTranslation(0.5, 0, 0);

// Union
const union = tf.booleanUnion(box0, box1);
console.log(`Union: ${union.mesh.numberOfFaces} faces`);
union.mesh.delete();
union.labels.delete();

// Intersection
const inter = tf.booleanIntersection(box0, box1);
console.log(`Intersection: ${inter.mesh.numberOfFaces} faces`);
inter.mesh.delete();
inter.labels.delete();

// Difference with curves
const diff = tf.booleanDifference(box0, box1, { returnCurves: true });
console.log(`Difference: ${diff.mesh.numberOfFaces} faces, ${diff.curves.length} curves`);
diff.mesh.delete();
diff.labels.delete();
diff.curves.points.delete();

box0.delete();
box1.delete();

// =========================================================================
// 5. Isocontours
// =========================================================================
console.log("\n" + "=".repeat(60));
console.log("=== Isocontours ===");
console.log("=".repeat(60));

// Create a plane mesh with scalar field
const plane = tf.planeMesh(2, 2, 20, 20);
console.log(`Plane: ${plane.numberOfFaces} faces, ${plane.numberOfPoints} points`);

// Scalar field: distance from origin along X axis
const scalars = plane.points.take(null, 0); // column 0 = X coords
console.log("Scalar range:", tf.min(scalars), "to", tf.max(scalars));

// Extract isocontours at multiple thresholds
const thresholds = new Float32Array([-0.5, 0.0, 0.5]);
const curves = tf.isocontours(plane, scalars, thresholds);
console.log(`Isocontours: ${curves.length} curves`);

// Isobands
const bands = tf.isobands(plane, scalars, thresholds);
console.log(`Isobands: ${bands.mesh.numberOfFaces} faces`);
bands.mesh.delete();
bands.labels.delete();

scalars.delete();
curves.points.delete();
plane.delete();

// =========================================================================
// 6. Transformations
// =========================================================================
console.log("\n" + "=".repeat(60));
console.log("=== Transformations ===");
console.log("=".repeat(60));

// Create two meshes
const meshA = tf.boxMesh(1, 1, 1);
const meshB = tf.boxMesh(1, 1, 1);
meshB.transformation = tf.makeTranslation(3, 0, 0);
meshA.buildTree();
meshB.buildTree();

// Initially no collision (3 units apart)
console.log("Initial collision:", tf.intersects(meshA, meshB));

// Move meshA close to meshB (tree is reused, no rebuild needed)
meshA.transformation = tf.makeTranslation(2.5, 0, 0);
console.log("After translation:", tf.intersects(meshA, meshB));

meshA.delete();
meshB.delete();

console.log("\nDone.");

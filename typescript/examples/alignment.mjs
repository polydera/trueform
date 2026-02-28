/**
 * Point cloud alignment example
 *
 * Demonstrates alignment between point clouds:
 * 1. With correspondences (smoothed mesh -> same vertex count)
 * 2. Without correspondences (shuffled points)
 * 3. ICP refinement (Point-to-Point vs Point-to-Plane)
 * 4. Different mesh resolutions
 *
 * Usage:
 *     node alignment.mjs [mesh.stl]
 *
 * Default mesh: ../../benchmarks/data/dragon-500k.stl
 */

import * as tf from "@polydera/trueform";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// =========================================================================
// Helpers
// =========================================================================

function randomTransformation(centroid, translation) {
  // Random rotation around centroid + far translation
  const R = tf.makeRandomRotation(centroid);
  const T = tf.makeTranslation(translation);
  const result = T.matMul(R);
  return result;
}

function transformPoints(points, T) {
  // points [N,3], T [4,4] → (points @ R^T) + t
  const rot = T.take([0, 1, 2], [0, 1, 2]); // [3,3]
  const trans = T.take([0, 1, 2], 3);         // [3]
  return points.matMul(rot.T).add(trans);      // [N,3]
}

function computeRmsError(A, B) {
  const diff = A.sub(B);
  return Math.sqrt(tf.mean(tf.sum(diff.mul(diff), 1)));
}

function computeMaxError(A, B) {
  const diff = A.sub(B);
  return Math.sqrt(tf.max(tf.sum(diff.mul(diff), 1)));
}

// =========================================================================
// Load mesh
// =========================================================================

const dataDir = resolve(__dirname, "../../benchmarks/data");
const meshPath = process.argv[2] || resolve(dataDir, "dragon-500k.stl");

console.log(`Loading mesh: ${meshPath}`);
const mesh = tf.readStl(readFileSync(meshPath));

if (mesh.numberOfFaces === 0) {
  console.log("Failed to load mesh or mesh is empty");
  process.exit(1);
}

const points = mesh.points;
console.log(`Loaded ${mesh.numberOfFaces} triangles, ${mesh.numberOfPoints} vertices`);

// Compute AABB diagonal
const aabbMin = tf.min(points, 0);
const aabbMax = tf.max(points, 0);
const diag = tf.norm(aabbMax.sub(aabbMin));
console.log(`AABB diagonal: ${diag.toFixed(2)}`);

// Smooth mesh to create source with known correspondences
const smoothIters = 200;
const smoothLambda = 0.9;
console.log(`\nSmoothing mesh (${smoothIters} iterations, lambda=${smoothLambda})...`);

const smoothed = tf.laplacianSmoothed(mesh, smoothIters, smoothLambda);
const smoothedPoints = smoothed.points;

const smoothRms = computeRmsError(points, smoothedPoints);
console.log(`Smoothing RMS displacement: ${smoothRms.toFixed(6)} (${(100 * smoothRms / diag).toFixed(2)}% of diagonal)`);

// Build target cloud
const targetCloud = tf.pointCloud(points);

// Compute point normals for point-to-plane ICP
console.log("Computing point normals...");
const targetNormals = mesh.pointNormals;
targetCloud.normals = targetNormals;

// =========================================================================
// Part 1: With correspondences
// =========================================================================
console.log("\n" + "=".repeat(60));
console.log("=== PART 1: With correspondences ===");
console.log("=".repeat(60));

const centroid = tf.mean(smoothedPoints, 0);
const farTranslation = [diag * 2.5, diag * -1.5, diag * 2.0];
const T1 = randomTransformation(centroid, farTranslation);

console.log("\nTransforming smoothed mesh (rotation around centroid + translation)");
const source1Pts = transformPoints(smoothedPoints, T1);
const initial1 = computeMaxError(points, source1Pts);
console.log(`Initial error: ${initial1.toFixed(2)}`);

const source1 = tf.pointCloud(source1Pts);
const baseline1 = tf.pointCloud(smoothedPoints);

console.log("\nRigid alignment:");
const TRigid1 = tf.fitRigidAlignment(source1, baseline1);
const rigid1Pts = transformPoints(source1Pts, TRigid1);
const rigid1Rms = computeRmsError(points, rigid1Pts);
console.log(`  RMS error: ${rigid1Rms.toFixed(6)}`);

console.log("\nOBB alignment (no tree):");
const baselineCloud = tf.pointCloud(points);
const TObb1NoTree = tf.fitObbAlignment(source1, baselineCloud, { sampleSize: 0 });
const obb1NoTreePts = transformPoints(source1Pts, TObb1NoTree);
const obb1NoTreeRms = computeRmsError(points, obb1NoTreePts);
console.log(`  RMS error: ${obb1NoTreeRms.toFixed(6)}`);

console.log("\nOBB alignment (with tree):");
const TObb1Tree = tf.fitObbAlignment(source1, targetCloud);
const obb1TreePts = transformPoints(source1Pts, TObb1Tree);
const obb1TreeRms = computeRmsError(points, obb1TreePts);
console.log(`  RMS error: ${obb1TreeRms.toFixed(6)}`);

console.log("\n--- Summary (Part 1) ---");
console.log(`  Ground truth:    ${smoothRms.toFixed(6)}`);
console.log(`  Rigid:           ${rigid1Rms.toFixed(6)}`);
console.log(`  OBB (no tree):   ${obb1NoTreeRms.toFixed(6)}`);
console.log(`  OBB (with tree): ${obb1TreeRms.toFixed(6)}`);

// =========================================================================
// Part 2: Without correspondences (shuffled)
// =========================================================================
console.log("\n" + "=".repeat(60));
console.log("=== PART 2: Without correspondences (shuffled) ===");
console.log("=".repeat(60));

// Shuffle source points
const nPts = source1Pts.shape[0];
const shuffleIds = tf.random("int32", [nPts], 0, nPts).argsort();
const source2Pts = source1Pts.take(shuffleIds, 0);
const targetShuffled = points.take(shuffleIds, 0);

const source2 = tf.pointCloud(source2Pts);

console.log("\nRigid alignment (will fail - no correspondences):");
const TRigid2 = tf.fitRigidAlignment(source2, baselineCloud);
const rigid2Pts = transformPoints(source2Pts, TRigid2);
const rigid2Rms = computeRmsError(targetShuffled, rigid2Pts);
console.log(`  RMS error: ${rigid2Rms.toFixed(6)}`);

console.log("\nOBB alignment (no tree - ambiguous):");
const TObb2NoTree = tf.fitObbAlignment(source2, baselineCloud, { sampleSize: 0 });
const obb2NoTreePts = transformPoints(source2Pts, TObb2NoTree);
const obb2NoTreeRms = computeRmsError(targetShuffled, obb2NoTreePts);
console.log(`  RMS error: ${obb2NoTreeRms.toFixed(6)}`);

console.log("\nOBB alignment (with tree - disambiguated):");
const TObb2Tree = tf.fitObbAlignment(source2, targetCloud);
const obb2TreePts = transformPoints(source2Pts, TObb2Tree);
const obb2TreeRms = computeRmsError(targetShuffled, obb2TreePts);
console.log(`  RMS error: ${obb2TreeRms.toFixed(6)}`);

console.log("\n--- Summary (Part 2) ---");
console.log(`  Ground truth:    ${smoothRms.toFixed(6)}`);
console.log(`  Rigid:           ${rigid2Rms.toFixed(6)} (FAILS)`);
console.log(`  OBB (no tree):   ${obb2NoTreeRms.toFixed(6)} (may be wrong orientation)`);
console.log(`  OBB (with tree): ${obb2TreeRms.toFixed(6)}`);

// =========================================================================
// Part 3: ICP refinement
// =========================================================================
console.log("\n" + "=".repeat(60));
console.log("=== PART 3: ICP refinement ===");
console.log("=".repeat(60));

console.log(`Ground truth RMS: ${smoothRms.toFixed(6)}`);
console.log(`Starting from OBB with tree: RMS = ${obb2TreeRms.toFixed(6)}`);

const maxIterations = 50;
const nSamples = 1000;
const k = 1;
const minRelativeImprovement = 1e-6;

console.log(`Subsampling: ~${nSamples} / ${nPts} points per iteration`);

// Point-to-Point ICP
console.log("\nPoint-to-Point ICP...");
source2.transformation = TObb2Tree;
let t0 = performance.now();
const Tp2pDelta = tf.fitIcpAlignment(source2, targetCloud, {
  maxIterations, nSamples, k, minRelativeImprovement,
});
const p2pTime = (performance.now() - t0).toFixed(1);
const Tp2p = Tp2pDelta.matMul(TObb2Tree);
const p2pPts = transformPoints(source2Pts, Tp2p);
const p2pRms = computeRmsError(targetShuffled, p2pPts);
console.log(`  Final RMS: ${p2pRms.toFixed(6)}, time: ${p2pTime} ms`);

// Point-to-Plane ICP
console.log("\nPoint-to-Plane ICP...");
source2.transformation = TObb2Tree;
t0 = performance.now();
const Tp2lDelta = tf.fitIcpAlignment(source2, targetCloud, {
  maxIterations, nSamples, k, minRelativeImprovement, sigma: 1,
});
const p2lTime = (performance.now() - t0).toFixed(1);
const Tp2l = Tp2lDelta.matMul(TObb2Tree);
const p2lPts = transformPoints(source2Pts, Tp2l);
const p2lRms = computeRmsError(targetShuffled, p2lPts);
console.log(`  Final RMS: ${p2lRms.toFixed(6)}, time: ${p2lTime} ms`);

console.log("\n--- ICP Comparison ---");
console.log(`  Ground truth RMS:    ${smoothRms.toFixed(6)}`);
console.log(`  Point-to-Point: RMS=${p2pRms.toFixed(6)}, time=${p2pTime} ms`);
console.log(`  Point-to-Plane: RMS=${p2lRms.toFixed(6)}, time=${p2lTime} ms`);

// =========================================================================
// Part 4: Different mesh resolutions
// =========================================================================
console.log("\n" + "=".repeat(60));
console.log("=== PART 4: Different mesh resolutions ===");
console.log("=".repeat(60));

const lowResPath = resolve(dataDir, "dragon-50k.stl");
console.log(`\nLoading low-res mesh: ${lowResPath}`);

let lowMesh;
try {
  lowMesh = tf.readStl(readFileSync(lowResPath));
  if (lowMesh.numberOfFaces === 0) throw new Error("empty");
} catch {
  console.log("Failed to load low-res mesh, skipping Part 4");
  process.exit(0);
}

const lowPoints = lowMesh.points;
console.log(`High-res: ${mesh.numberOfPoints} vertices`);
console.log(`Low-res:  ${lowMesh.numberOfPoints} vertices`);

const lowCloud = tf.pointCloud(lowPoints);

// Baseline Chamfer (aligned, different resolutions)
const chamferBaselineFwd = tf.chamferError(lowCloud, targetCloud);
const chamferBaselineBwd = tf.chamferError(targetCloud, lowCloud);
console.log("\nBaseline Chamfer (aligned, different resolutions):");
console.log(`  Low->High: ${chamferBaselineFwd.toFixed(6)}`);
console.log(`  High->Low: ${chamferBaselineBwd.toFixed(6)}`);
console.log(`  Symmetric: ${((chamferBaselineFwd + chamferBaselineBwd) / 2).toFixed(6)}`);

// Transform low-res mesh far away
const centroidLow = tf.mean(lowPoints, 0);
const TLow = randomTransformation(centroidLow, farTranslation);
const sourceLowPts = transformPoints(lowPoints, TLow);
const sourceLow = tf.pointCloud(sourceLowPts);

// Initial Chamfer
const chamferInitFwd = tf.chamferError(sourceLow, targetCloud);
console.log(`\nInitial Chamfer: ${chamferInitFwd.toFixed(2)}`);

// OBB (no tree)
console.log("\nOBB alignment (no tree):");
const TObbLowNoTree = tf.fitObbAlignment(sourceLow, baselineCloud, { sampleSize: 0 });
sourceLow.transformation = TObbLowNoTree;
const chamferObbNoTree = tf.chamferError(sourceLow, targetCloud);
console.log(`  Chamfer (Low->High): ${chamferObbNoTree.toFixed(6)}`);
sourceLow.transformation = null;

// OBB (with tree)
console.log("\nOBB alignment (with tree):");
const TObbLowTree = tf.fitObbAlignment(sourceLow, targetCloud);
sourceLow.transformation = TObbLowTree;
const chamferObbTree = tf.chamferError(sourceLow, targetCloud);
console.log(`  Chamfer (Low->High): ${chamferObbTree.toFixed(6)}`);

// ICP P2P
console.log("\nPoint-to-Point ICP...");
sourceLow.transformation = TObbLowTree;
t0 = performance.now();
const Tp2pLowDelta = tf.fitIcpAlignment(sourceLow, targetCloud, {
  maxIterations, nSamples, k, minRelativeImprovement,
});
const p2pTimeLow = (performance.now() - t0).toFixed(1);
const Tp2pLow = Tp2pLowDelta.matMul(TObbLowTree);
sourceLow.transformation = Tp2pLow;
const p2pChamferLow = tf.chamferError(sourceLow, targetCloud);
console.log(`  Chamfer: ${p2pChamferLow.toFixed(6)}, time: ${p2pTimeLow} ms`);

// ICP P2L
console.log("\nPoint-to-Plane ICP...");
sourceLow.transformation = TObbLowTree;
t0 = performance.now();
const Tp2lLowDelta = tf.fitIcpAlignment(sourceLow, targetCloud, {
  maxIterations, nSamples, k, minRelativeImprovement, sigma: 1,
});
const p2lTimeLow = (performance.now() - t0).toFixed(1);
const Tp2lLow = Tp2lLowDelta.matMul(TObbLowTree);
sourceLow.transformation = Tp2lLow;
const p2lChamferLow = tf.chamferError(sourceLow, targetCloud);
console.log(`  Chamfer: ${p2lChamferLow.toFixed(6)}, time: ${p2lTimeLow} ms`);

console.log("\n--- Summary (Part 4) ---");
console.log(`  Baseline:        ${chamferBaselineFwd.toFixed(6)} (best possible)`);
console.log(`  Initial:         ${chamferInitFwd.toFixed(2)} (after transformation)`);
console.log(`  OBB (no tree):   ${chamferObbNoTree.toFixed(6)}`);
console.log(`  OBB (with tree): ${chamferObbTree.toFixed(6)}`);
console.log(`  P2P ICP: Chamfer=${p2pChamferLow.toFixed(6)}, time=${p2pTimeLow} ms`);
console.log(`  P2L ICP: Chamfer=${p2lChamferLow.toFixed(6)}, time=${p2lTimeLow} ms`);

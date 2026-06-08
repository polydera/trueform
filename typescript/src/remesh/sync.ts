/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */

import { native } from "../native";
import { Mesh } from "../form/Mesh";
import { ndarray } from "../ndarray/factories";
import { NDArray, type NDArrayInt32 } from "../ndarray/NDArray";

/** Per-face region labels passed to a region-preserving remesh. */
export type RegionsInput = Int32Array | number[] | NDArrayInt32;

/** A remesh result that also carries the output mesh's per-face region labels. */
export interface RemeshWithRegions {
  mesh: Mesh;
  regions: NDArrayInt32;
}

/** Build the native int32 region-label handle. When no regions are requested
 *  (preserveRegions omitted) an empty array is passed so the native side runs
 *  the plain path. A supplied array must have exactly one label per face, else
 *  the WASM side would read out of bounds -- so it is validated here. */
export function _regionsNd(preserveRegions: RegionsInput | undefined,
  nFaces: number): NDArray {
  if (preserveRegions == null) return ndarray(new Int32Array(0));
  const len = preserveRegions.length;
  if (len !== nFaces)
    throw new Error(
      `preserveRegions must have one label per face (${nFaces}), got ${len}`,
    );
  if (preserveRegions instanceof NDArray) return preserveRegions;
  const arr = preserveRegions instanceof Int32Array
    ? preserveRegions
    : Int32Array.from(preserveRegions);
  return ndarray(arr);
}

/** Options for quadric error metric decimation. */
export interface DecimateOptions {
  /** Worst triangle quality allowed after a collapse, in [0,1] (1 = equilateral).
   *  Negative disables (default), 0 = never worsen, >0 = quality floor. Default: -1 */
  minQuality?: number;
  /** If true, boundary edges are never collapsed. Default: true */
  preserveBoundary?: boolean;
  /** Tikhonov stabilizer for quadric solve. Default: 1e-3 */
  stabilizer?: number;
  /** Use parallel partitioned collapse. Default: true */
  parallel?: boolean;
  /** Feature edge detection angle in degrees. Edges sharper than this are preserved. Negative disables. Default: -1 */
  featureAngle?: number;
  /** Feature edge preservation weight. Higher = stronger. Default: 100 */
  featureWeight?: number;
  /** Per-face region labels (one int per input face). When given, edges between
   *  differing labels are preserved as features and the call returns
   *  { mesh, regions } (the output mesh's per-face labels) instead of a Mesh. */
  preserveRegions?: RegionsInput;
}

/** Options for isotropic remeshing. */
export interface RemeshOptions {
  /** Number of outer iterations (split + collapse + flip + relax). Default: 3 */
  iterations?: number;
  /** Tangential relaxation iterations per outer iteration. Default: 3 */
  relaxationIters?: number;
  /** Worst triangle quality allowed after a collapse, in [0,1] (1 = equilateral).
   *  Negative disables, 0 = never worsen, >0 = quality floor. Default: 0.3 */
  minQuality?: number;
  /** Damping factor for tangential relaxation in (0, 1]. Default: 0.5 */
  lambda?: number;
  /** If true, boundary edges are never split or collapsed. Default: true */
  preserveBoundary?: boolean;
  /** Use quadric error metric for collapse vertex placement. Default: false */
  useQuadric?: boolean;
  /** Use parallel execution. Default: true */
  parallel?: boolean;
  /** Feature edge detection angle in degrees. Edges sharper than this are preserved. Negative disables. Default: -1 */
  featureAngle?: number;
  /** Feature edge preservation weight. Higher = stronger. Default: 100 */
  featureWeight?: number;
  /** Per-face region labels (one int per input face). When given, edges between
   *  differing labels are preserved as features and the call returns
   *  { mesh, regions } (the output mesh's per-face labels) instead of a Mesh. */
  preserveRegions?: RegionsInput;
}

/** Decimate a mesh to a target proportion of original faces. Returns a new Mesh,
 *  or { mesh, regions } when preserveRegions is given. */
export function decimated(m: Mesh, targetProportion: number, opts?: DecimateOptions): Mesh;
export function decimated(m: Mesh, targetProportion: number,
  opts: DecimateOptions & { preserveRegions: RegionsInput }): RemeshWithRegions;
export function decimated(m: Mesh, targetProportion: number,
  opts?: DecimateOptions): Mesh | RemeshWithRegions {
  const fa = opts?.featureAngle != null && opts.featureAngle >= 0
    ? opts.featureAngle * Math.PI / 180 : -1;
  const dt = m.dtype;
  const regions = _regionsNd(opts?.preserveRegions, m.numberOfFaces);
  const raw = native()[`decimated_${dt}`](
    m._handle,
    targetProportion,
    opts?.minQuality ?? -1,
    opts?.preserveBoundary ?? true,
    opts?.stabilizer ?? 1e-3,
    opts?.parallel ?? true,
    fa,
    opts?.featureWeight ?? 100,
    regions._handle,
  );
  const mesh = new Mesh(raw.mesh, dt);
  const outRegions = new NDArray<Int32Array>(raw.regions, "int32");
  return opts?.preserveRegions != null ? { mesh, regions: outRegions } : mesh;
}

/** Options for error-budget simplification. */
export interface SimplifyOptions {
  /** Error budget as a fraction of the mesh bounding-box diagonal. An edge is
   *  collapsed when its quadric error is <= errorRel * diagonal. Default: 0.002 */
  errorRel?: number;
  /** Quality cleanup after each collapse: this many rounds of min-angle edge
   *  flip + tangential relaxation. 0 = pure error-budget collapse. Default: 3 */
  optimizeIterations?: number;
  /** Outer collapse rounds. 1 = single collapse + cleanup (plain simplify);
   *  >1 re-collapses after each cleanup (iterated remesh), removing more at the
   *  cost of more deviation from the original. Default: 1 */
  iterations?: number;
  /** Tangential relaxation passes per cleanup round. Default: 3 */
  relaxationIters?: number;
  /** Damping factor for tangential relaxation in (0, 1]. Default: 0.5 */
  lambda?: number;
  /** Worst triangle quality allowed after a collapse, in [0,1] (1 = equilateral).
   *  Negative disables, 0 = never worsen, >0 = quality floor. Default: 0.3 */
  minQuality?: number;
  /** If true, boundary edges are never collapsed. Default: true */
  preserveBoundary?: boolean;
  /** Tikhonov stabilizer for quadric solve. Default: 1e-3 */
  stabilizer?: number;
  /** Use parallel partitioned collapse. Default: true */
  parallel?: boolean;
  /** Feature edge detection angle in degrees. Edges sharper than this are preserved. Negative disables. Default: -1 */
  featureAngle?: number;
  /** Feature edge preservation weight. Higher = stronger. Default: 100 */
  featureWeight?: number;
  /** Per-face region labels (one int per input face). When given, edges between
   *  differing labels are preserved as features and the call returns
   *  { mesh, regions } (the output mesh's per-face labels) instead of a Mesh. */
  preserveRegions?: RegionsInput;
}

/** Isotropic remesh to uniform target edge length. Returns a new Mesh, or
 *  { mesh, regions } when preserveRegions is given. */
export function isotropicRemeshed(m: Mesh, targetLength: number, opts?: RemeshOptions): Mesh;
export function isotropicRemeshed(m: Mesh, targetLength: number,
  opts: RemeshOptions & { preserveRegions: RegionsInput }): RemeshWithRegions;
export function isotropicRemeshed(m: Mesh, targetLength: number,
  opts?: RemeshOptions): Mesh | RemeshWithRegions {
  const fa = opts?.featureAngle != null && opts.featureAngle >= 0
    ? opts.featureAngle * Math.PI / 180 : -1;
  const dt = m.dtype;
  const regions = _regionsNd(opts?.preserveRegions, m.numberOfFaces);
  const raw = native()[`isotropic_remeshed_${dt}`](
    m._handle,
    targetLength,
    opts?.iterations ?? 3,
    opts?.relaxationIters ?? 3,
    opts?.minQuality ?? 0.3,
    opts?.lambda ?? 0.5,
    opts?.preserveBoundary ?? true,
    opts?.useQuadric ?? false,
    opts?.parallel ?? true,
    fa,
    opts?.featureWeight ?? 100,
    regions._handle,
  );
  const mesh = new Mesh(raw.mesh, dt);
  const outRegions = new NDArray<Int32Array>(raw.regions, "int32");
  return opts?.preserveRegions != null ? { mesh, regions: outRegions } : mesh;
}

/** Simplify a mesh to an error budget using quadric error metrics. Returns a new
 *  Mesh, or { mesh, regions } when preserveRegions is given. */
export function simplified(m: Mesh, opts?: SimplifyOptions): Mesh;
export function simplified(m: Mesh,
  opts: SimplifyOptions & { preserveRegions: RegionsInput }): RemeshWithRegions;
export function simplified(m: Mesh, opts?: SimplifyOptions): Mesh | RemeshWithRegions {
  const fa = opts?.featureAngle != null && opts.featureAngle >= 0
    ? opts.featureAngle * Math.PI / 180 : -1;
  const dt = m.dtype;
  const regions = _regionsNd(opts?.preserveRegions, m.numberOfFaces);
  const raw = native()[`simplified_${dt}`](
    m._handle,
    opts?.errorRel ?? 0.002,
    opts?.optimizeIterations ?? 3,
    opts?.minQuality ?? 0.3,
    opts?.preserveBoundary ?? true,
    opts?.stabilizer ?? 1e-3,
    opts?.parallel ?? true,
    fa,
    opts?.featureWeight ?? 100,
    opts?.iterations ?? 1,
    opts?.relaxationIters ?? 3,
    opts?.lambda ?? 0.5,
    regions._handle,
  );
  const mesh = new Mesh(raw.mesh, dt);
  const outRegions = new NDArray<Int32Array>(raw.regions, "int32");
  return opts?.preserveRegions != null ? { mesh, regions: outRegions } : mesh;
}

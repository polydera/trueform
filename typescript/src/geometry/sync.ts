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
import { NDArray, type NDArrayFloat32, type NDArrayFloat64, type NDArrayInt32 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";
import { Curves } from "../form/Curves";
import { Mesh } from "../form/Mesh";
import { PointCloud } from "../form/PointCloud";
import { Primitive, Polygon } from "../primitive";
import { assertSameDtype } from "../internal/dtype";

export type { MeshLike } from "../form/MeshLike";
import type { MeshLike } from "../form/MeshLike";

/** Triangulate polygons into a triangle mesh. */
export function triangulate(input: Mesh | MeshLike | Polygon): Mesh {
  if (input instanceof Mesh) {
    const dt = input.dtype;
    return new Mesh(native()[`triangulate_mesh_${dt}`](input._handle), dt);
  }
  if (input instanceof Primitive) {
    const dt = input.dtype === "float64" ? "float64" : "float32";
    return new Mesh(native()[`triangulate_polygon_${dt}`](input._handle), dt);
  }
  // MeshLike
  const dt = input.points.dtype === "float64" ? "float64" : "float32";
  if (input.faces instanceof OffsetBlockedBuffer) {
    return new Mesh(native()[`triangulate_dynamic_${dt}`](input.faces._handle, input.points._handle), dt);
  }
  return new Mesh(native()[`triangulate_fixed_${dt}`](input.faces._handle, input.points._handle), dt);
}

/** Options selecting the coordinate dtype of returned mesh primitives. */
export interface DtypeOptions {
  dtype?: "float32" | "float64";
}

/** Create a UV sphere mesh centered at origin. */
export function sphereMesh(radius: number, stacks: number, segments: number, opts?: DtypeOptions): Mesh {
  const dt = opts?.dtype ?? "float32";
  return new Mesh(native()[`make_sphere_mesh_${dt}`](radius, stacks, segments), dt);
}

/** Create a cylinder mesh centered at origin along the z-axis. */
export function cylinderMesh(radius: number, height: number, segments: number, opts?: DtypeOptions): Mesh {
  const dt = opts?.dtype ?? "float32";
  return new Mesh(native()[`make_cylinder_mesh_${dt}`](radius, height, segments), dt);
}

/** Create an axis-aligned box mesh centered at origin. Optionally subdivided. */
export function boxMesh(
  width: number, height: number, depth: number,
  widthTicks?: number, heightTicks?: number, depthTicks?: number,
  opts?: DtypeOptions,
): Mesh {
  const dt = opts?.dtype ?? "float32";
  if (widthTicks !== undefined && heightTicks !== undefined && depthTicks !== undefined) {
    return new Mesh(native()[`make_box_mesh_subdivided_${dt}`](width, height, depth, widthTicks, heightTicks, depthTicks), dt);
  }
  return new Mesh(native()[`make_box_mesh_${dt}`](width, height, depth), dt);
}

/** Create a tube mesh from curves using parallel transport frames. Inherits dtype from the input curves. */
export function tubeMesh(curves: Curves, radius: number, radialSegments: number = 8): Mesh {
  const dt = curves.dtype;
  return new Mesh(native()[`make_tube_mesh_${dt}`](curves._handle, radius, radialSegments), dt);
}

/** Create a flat rectangular plane mesh in the XY plane, centered at origin. */
export function planeMesh(
  width: number, height: number,
  widthTicks?: number, heightTicks?: number,
  opts?: DtypeOptions,
): Mesh {
  const wt = widthTicks ?? 1;
  const ht = heightTicks ?? 1;
  const dt = opts?.dtype ?? "float32";
  return new Mesh(native()[`make_plane_mesh_${dt}`](width, height, wt, ht), dt);
}

// ============ Edge Analysis ============

/** Sharp edges where the dihedral angle exceeds the threshold (in degrees). Returns [N, 2] Int32 array of vertex index pairs. */
export function sharpEdges(m: Mesh, angleDeg: number): NDArrayInt32 {
  const dt = m.dtype;
  return new NDArray(native()[`sharp_edges_${dt}`](m._handle, angleDeg), "int32");
}

/** Every edge two faces share and the angle it turns through. */
export interface DihedralAnglesResult {
  /** [N, 2] int32 vertex index pairs, one per undirected edge two faces
   *  share. A boundary or non-manifold edge joins no pair and is not
   *  stated. */
  edges: NDArrayInt32;
  /** [N] radians between the face normals, aligned with `edges`; a flat
   *  surface reads 0. Dtype follows the mesh. */
  angles: NDArrayFloat32 | NDArrayFloat64;
}

/** Measure every edge two faces of the mesh share. */
export function dihedralAngles(m: Mesh): DihedralAnglesResult {
  const dt = m.dtype;
  const raw = native()[`dihedral_angles_${dt}`](m._handle);
  return {
    edges: new NDArray(raw.edges, "int32"),
    angles: new NDArray(raw.angles, dt) as NDArrayFloat32 | NDArrayFloat64,
  };
}

// ============ Face Quality ============

/** One array per measure, each of shape [F] in the mesh's dtype. */
export interface FaceQualityResult {
  /** The triangle quality measure: 1 for equilateral, approaching 0 for a
   *  sliver; a face that is not a triangle reads -1. */
  quality: NDArrayFloat32 | NDArrayFloat64;
  /** Smallest corner angle in radians. An angle is unsigned and lies in
   *  [0, pi], so a reflex corner reads its explement. */
  minAngle: NDArrayFloat32 | NDArrayFloat64;
  /** Largest corner angle in radians. */
  maxAngle: NDArrayFloat32 | NDArrayFloat64;
  /** Longest side over the shortest, infinite where a side has no length. */
  aspectRatio: NDArrayFloat32 | NDArrayFloat64;
}

/** Measure every face of the mesh. */
export function faceQuality(m: Mesh): FaceQualityResult {
  const dt = m.dtype;
  const raw = native()[`face_quality_${dt}`](m._handle);
  return {
    quality: new NDArray(raw.quality, dt) as NDArrayFloat32 | NDArrayFloat64,
    minAngle: new NDArray(raw.minAngle, dt) as NDArrayFloat32 | NDArrayFloat64,
    maxAngle: new NDArray(raw.maxAngle, dt) as NDArrayFloat32 | NDArrayFloat64,
    aspectRatio: new NDArray(raw.aspectRatio, dt) as NDArrayFloat32 | NDArrayFloat64,
  };
}

// ============ Measurements ============

/** Total surface area of a mesh. Respects transformation. */
export function area(m: Mesh): number {
  return native()[`mesh_area_${m.dtype}`](m._handle);
}

/** Signed volume of a closed 3D mesh. Positive for outward normals. Respects transformation. */
export function signedVolume(m: Mesh): number {
  return native()[`mesh_signed_volume_${m.dtype}`](m._handle);
}

/** Absolute volume of a closed 3D mesh. Respects transformation. */
export function volume(m: Mesh): number {
  return Math.abs(native()[`mesh_signed_volume_${m.dtype}`](m._handle));
}

/** Mean edge length across all faces. Respects transformation. */
export function meanEdgeLength(m: Mesh): number {
  return native()[`mesh_mean_edge_length_${m.dtype}`](m._handle);
}

/** Minimum edge length across all faces. Respects transformation. */
export function minEdgeLength(m: Mesh): number {
  return native()[`mesh_min_edge_length_${m.dtype}`](m._handle);
}

/** Maximum edge length across all faces. Respects transformation. */
export function maxEdgeLength(m: Mesh): number {
  return native()[`mesh_max_edge_length_${m.dtype}`](m._handle);
}

// ============ Orientation ============

/** Return a new mesh with reversed face winding (flipped normals). */
export function reverseWinding(m: Mesh): Mesh {
  const dt = m.dtype;
  return new Mesh(native()[`reverse_winding_${dt}`](m._handle), dt);
}

/** Return a new mesh with consistently oriented, outward-pointing normals. */
export function positivelyOriented(m: Mesh, isConsistent?: boolean): Mesh {
  const dt = m.dtype;
  return new Mesh(native()[`positively_oriented_${dt}`](m._handle, isConsistent ?? false), dt);
}

// ============ Curvature ============

/** Principal curvature result: k0 (max) and k1 (min) per vertex. */
export interface PrincipalCurvatures {
  k0: NDArrayFloat32 | NDArrayFloat64;
  k1: NDArrayFloat32 | NDArrayFloat64;
}

/** Principal curvature result with directions. */
export interface PrincipalDirections {
  k0: NDArrayFloat32 | NDArrayFloat64;
  k1: NDArrayFloat32 | NDArrayFloat64;
  d0: NDArrayFloat32 | NDArrayFloat64;
  d1: NDArrayFloat32 | NDArrayFloat64;
}

/** Compute principal curvatures (k0, k1) at each vertex. */
export function principalCurvatures(m: Mesh, k: number = 2): PrincipalCurvatures {
  const dt = m.dtype;
  const raw = native()[`principal_curvatures_${dt}`](m._handle, k);
  return {
    k0: new NDArray(raw.k0, dt) as NDArrayFloat32 | NDArrayFloat64,
    k1: new NDArray(raw.k1, dt) as NDArrayFloat32 | NDArrayFloat64,
  };
}

/** Compute principal curvatures and directions at each vertex. */
export function principalDirections(m: Mesh, k: number = 2): PrincipalDirections {
  const dt = m.dtype;
  const raw = native()[`principal_directions_${dt}`](m._handle, k);
  return {
    k0: new NDArray(raw.k0, dt) as NDArrayFloat32 | NDArrayFloat64,
    k1: new NDArray(raw.k1, dt) as NDArrayFloat32 | NDArrayFloat64,
    d0: new NDArray(raw.d0, dt) as NDArrayFloat32 | NDArrayFloat64,
    d1: new NDArray(raw.d1, dt) as NDArrayFloat32 | NDArrayFloat64,
  };
}

/** Compute shape index at each vertex. Values in [-1, 1]. */
export function shapeIndex(m: Mesh, k: number = 2): NDArrayFloat32 | NDArrayFloat64 {
  const dt = m.dtype;
  return new NDArray(native()[`shape_index_${dt}`](m._handle, k), dt) as NDArrayFloat32 | NDArrayFloat64;
}

// ============ Smoothing ============

/** Laplacian smoothing. Returns a new mesh with smoothed vertex positions. */
export function laplacianSmoothed(m: Mesh, iterations: number, lambda: number = 0.5): Mesh {
  const dt = m.dtype;
  return new Mesh(native()[`laplacian_smoothed_${dt}`](m._handle, iterations, lambda), dt);
}

/** Taubin smoothing (volume-preserving). Returns a new mesh with smoothed vertex positions. */
export function taubinSmoothed(m: Mesh, iterations: number, lambda: number = 0.5, kpb: number = 0.1): Mesh {
  const dt = m.dtype;
  return new Mesh(native()[`taubin_smoothed_${dt}`](m._handle, iterations, lambda, kpb), dt);
}

// ============ Surface fields ============

/** Options for {@link geodesicDistances}. */
export interface GeodesicDistancesOptions {
  /** The reach of the front (default: Infinity). */
  radius?: number;
}

/** Options for {@link dirichletField}. */
export interface DirichletFieldOptions {
  /** `3 * numberOfFaces` weights; entry `3 f + c` weighs the edge of
   *  face `f` opposite its corner `c`, summed over the faces an edge borders.
   *  Absent, the cotangent weights stand. */
  cornerWeights?: NDArrayFloat64;
}

/** The operator a {@link dirichletField} vanishes under (C++
 *  `tf::dirichlet_order`): `harmonic` is `L x = 0`, `biharmonic` is
 *  `L M^-1 L x = 0`. */
export type DirichletOrder = "harmonic" | "biharmonic";

/** What a {@link dirichletField} solve answered (C++
 *  `tf::dirichlet_field_status`). `refusedOperator` is a corner weight that
 *  is not finite, or a point whose lumped mass is not positive — one no face
 *  has as a corner, or one whose every face is degenerate;
 *  `refusedUnanchored` a connected run of free points reaching no fixed one;
 *  `refusedIndex` an index type too narrow to name the factor;
 *  `refusedIndefinite` a pivot that is not positive; `refusedResidual` an
 *  answer whose true residual misses its acceptance. */
export const DirichletFieldStatus = {
  solved: 0,
  refusedOperator: 1,
  refusedUnanchored: 2,
  refusedIndex: 3,
  refusedIndefinite: 4,
  refusedResidual: 5,
} as const;

/** What {@link dirichletField} solved (C++ `tf::dirichlet_field_result`). */
export interface DirichletField {
  /** [N] one value per point, in the mesh's dtype. The fixed values stand at
   *  the fixed points whatever happened; a solved field states its answer
   *  everywhere else, and a refused one states zero there. */
  field: NDArrayFloat32 | NDArrayFloat64;
  /** A {@link DirichletFieldStatus} value. */
  status: number;
}

const DIRICHLET_ORDERS: Record<DirichletOrder, number> = { harmonic: 0, biharmonic: 1 };

function requirePointIds(ids: NDArrayInt32, n: number, name: string, distinct: boolean): void {
  if (!(ids instanceof NDArray) || ids.dtype !== "int32")
    throw new TypeError(`${name} must be an int32 NDArray`);
  if (ids.shape.length !== 1) throw new RangeError(`${name} must be one-dimensional`);
  const data = ids.data as Int32Array;
  const seen = distinct ? new Uint8Array(n) : null;
  for (let k = 0; k < data.length; k++) {
    const id = data[k]!;
    if (id < 0 || id >= n) throw new RangeError(`${name}: point id out of range`);
    if (seen) {
      if (seen[id]) throw new RangeError(`${name}: point ids must be distinct`);
      seen[id] = 1;
    }
  }
}

function requireValues(values: NDArrayFloat64, length: number, what: string): void {
  if (!(values instanceof NDArray) || values.dtype !== "float64")
    throw new TypeError(`dirichletField: ${what} must be a float64 NDArray`);
  if (values.shape.length !== 1 || values.length !== length)
    throw new RangeError(`dirichletField: ${what} must hold ${length} values`);
}

/** @internal */
export function geodesicArgs(m: Mesh, seeds: NDArrayInt32, opts?: GeodesicDistancesOptions): unknown[] {
  requirePointIds(seeds, m.numberOfPoints, "seeds", false);
  const radius = opts?.radius ?? Infinity;
  if (Number.isNaN(radius) || radius < 0)
    throw new RangeError("geodesicDistances: radius must be non-negative and not NaN");
  return [m._handle, seeds._handle, radius];
}

/** @internal The native name's stem and its arguments. */
export function dirichletArgs(
  m: Mesh, fixedIds: NDArrayInt32, fixedValues: NDArrayFloat64, order: DirichletOrder,
  opts?: DirichletFieldOptions,
): [string, unknown[]] {
  requirePointIds(fixedIds, m.numberOfPoints, "fixedIds", true);
  requireValues(fixedValues, fixedIds.length, "fixedValues");
  const data = fixedValues.data as Float64Array;
  for (let k = 0; k < data.length; k++)
    if (!Number.isFinite(data[k])) throw new RangeError("dirichletField: fixedValues must be finite");
  const stated = DIRICHLET_ORDERS[order];
  if (stated === undefined)
    throw new RangeError(`dirichletField: order must be "harmonic" or "biharmonic", got "${order}"`);
  const ids = [fixedIds._handle, fixedValues._handle, stated];
  if (opts?.cornerWeights === undefined) return ["dirichlet_field_cotangent", [m._handle, ...ids]];
  requireValues(opts.cornerWeights, 3 * m.numberOfFaces, "cornerWeights");
  return ["dirichlet_field", [m._handle, opts.cornerWeights._handle, ...ids]];
}

/** @internal */
export function wrapDirichletField(raw: any, dtype: Mesh["dtype"]): DirichletField {
  return { field: new NDArray(raw.field, dtype) as NDArrayFloat32 | NDArrayFloat64, status: raw.status };
}

/**
 * The geodesic distance from the nearest seed to every point.
 *
 * A label-correcting front with the two-neighbour triangle-unfolding update:
 * a seed's ring is reached at its Euclidean distance, and the front runs from
 * there. On a planar mesh it is the straight line; on a curved one it beats
 * the shortest edge path. Measured in the stored coordinates; the mesh's
 * transformation does not enter.
 *
 * @param m - The mesh
 * @param seeds - [S] the point ids the distance is measured from
 * @returns [N] in the mesh's dtype: zero at a seed, Infinity where no front
 *   reaches — another piece of the mesh, or past `radius` — and at most
 *   `radius` otherwise
 */
export function geodesicDistances(
  m: Mesh, seeds: NDArrayInt32, opts?: GeodesicDistancesOptions,
): NDArrayFloat32 | NDArrayFloat64 {
  const args = geodesicArgs(m, seeds, opts);
  return new NDArray(native()[`geodesic_distances_${m.dtype}`](...args), m.dtype) as
    NDArrayFloat32 | NDArrayFloat64;
}

/**
 * The field held at `fixedIds` and harmonic (`L x = 0`) or biharmonic
 * (`L M^-1 L x = 0`) at every other point, where `L` is the Laplacian the
 * corner weights state and `M` the barycentric lumped mass. The free rows are
 * solved by sparse Cholesky, refined once and accepted against their true
 * residual; the result's `status` names what refused. Read in the stored
 * coordinates.
 *
 * @param m - A triangle mesh
 * @param fixedIds - [K] the distinct point ids the field is held at
 * @param fixedValues - [K] the finite value at each of them
 * @param order - `"harmonic"` or `"biharmonic"`
 */
export function dirichletField(
  m: Mesh, fixedIds: NDArrayInt32, fixedValues: NDArrayFloat64, order: DirichletOrder,
  opts?: DirichletFieldOptions,
): DirichletField {
  const [stem, args] = dirichletArgs(m, fixedIds, fixedValues, order, opts);
  return wrapDirichletField(native()[`${stem}_${m.dtype}`](...args), m.dtype);
}

// ============ Registration / Alignment ============

/** Options for ICP alignment. */
export interface IcpOptions {
  maxIterations?: number;
  nSamples?: number;
  k?: number;
  sigma?: number;
  outlierProportion?: number;
  minRelativeImprovement?: number;
  emaAlpha?: number;
}

/** Options for OBB alignment. */
export interface ObbOptions {
  sampleSize?: number;
}

/** Options for Chamfer error. */
export interface ChamferOptions {
  outlierProportion?: number;
}

/** Kabsch/SVD rigid alignment. Requires 1:1 correspondence (|source| == |target|). Returns 4x4 delta matrix. */
export function fitRigidAlignment(source: PointCloud, target: PointCloud): NDArrayFloat32 | NDArrayFloat64 {
  assertSameDtype([source, target], ["source", "target"]);
  const dt = source.dtype;
  return new NDArray(native()[`fit_rigid_alignment_${dt}`](source._handle, target._handle), dt) as NDArrayFloat32 | NDArrayFloat64;
}

/** Iterative Closest Point alignment. Returns 4x4 delta matrix. */
export function fitIcpAlignment(source: PointCloud, target: PointCloud, opts?: IcpOptions): NDArrayFloat32 | NDArrayFloat64 {
  assertSameDtype([source, target], ["source", "target"]);
  const dt = source.dtype;
  return new NDArray(native()[`fit_icp_alignment_${dt}`](
    source._handle, target._handle,
    opts?.maxIterations ?? 100,
    opts?.nSamples ?? 1000,
    opts?.k ?? 1,
    opts?.sigma ?? -1,
    opts?.outlierProportion ?? 0,
    opts?.minRelativeImprovement ?? 1e-6,
    opts?.emaAlpha ?? 0.3,
  ), dt) as NDArrayFloat32 | NDArrayFloat64;
}

/** Coarse alignment via oriented bounding boxes. Returns 4x4 delta matrix. */
export function fitObbAlignment(source: PointCloud, target: PointCloud, opts?: ObbOptions): NDArrayFloat32 | NDArrayFloat64 {
  assertSameDtype([source, target], ["source", "target"]);
  const dt = source.dtype;
  return new NDArray(native()[`fit_obb_alignment_${dt}`](
    source._handle, target._handle,
    opts?.sampleSize ?? 100,
  ), dt) as NDArrayFloat32 | NDArrayFloat64;
}

/** One-way Chamfer distance (mean nearest-neighbor distance from source to target). */
export function chamferError(source: PointCloud, target: PointCloud, opts?: ChamferOptions): number {
  assertSameDtype([source, target], ["source", "target"]);
  const dt = source.dtype;
  return native()[`chamfer_error_${dt}`](
    source._handle, target._handle,
    opts?.outlierProportion ?? 0,
  );
}

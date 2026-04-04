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

// Auto-initialize WASM on import
import { init } from "./native";
await init();

// Geometry primitives (WASM-resident, extends NDArray)
export {
  Primitive, PrimitiveType,
  Point, Vector, Segment, Triangle,
  Ray, Line, Plane, AABB, Polygon,
} from "./primitive";

// NDArray + typed aliases
export { NDArray, NDArrayInt8, NDArrayInt32, NDArrayFloat32, NDArrayBool } from "./ndarray/NDArray";
export { OffsetBlockedBuffer } from "./ndarray/OffsetBlockedBuffer";
export { IndexMap } from "./core/IndexMap";
export { Mesh } from "./form/Mesh";
export { PointCloud } from "./form/PointCloud";
export { Curves } from "./form/Curves";

// NDArray factories
export {
  ndarray, random, stack, concatenate, tile,
  where, zeros, ones, full, eye, arange, linspace,
  take, takeAlongAxis, sort, sort_, argsort,
  unique, setUnion, setIntersection, setDifference,
} from "./ndarray/factories";

// Mesh, PointCloud & Curves factories
export { mesh, pointCloud, curves } from "./form/factories";

// Primitive factories
export {
  point, vector, segment, triangle,
  ray, line, plane, aabb, aabbFrom, polygon,
} from "./primitive";

// Transformations
export { makeTranslation, makeRotation, makeRandomRotation } from "./primitive";
export type { Axis } from "./primitive";

// Sync reductions
export { sum, min, max, mean, norm, argmin, argmax, any, all } from "./ndarray/reductions";

// Math functions
export {
  sqrt, sqrt_, sin, sin_, cos, cos_, tan, tan_,
  asin, asin_, acos, acos_, atan, atan_,
  exp, exp_, log, log_, log2, log2_, log10, log10_,
  floor, floor_, ceil, ceil_, round, round_,
  pow, pow_, abs, neg, clip,
  dot, cross, atan2,
  normalize, normalize_,
  mod, mod_,
  isNaN,
} from "./ndarray/math";

// IO
export { readStl, readStlData, readObj, readObjData, writeStl, writeObj } from "./io/sync";
export type { ReadObjOptions } from "./io/sync";

// Cut operations
export {
  booleanUnion, booleanIntersection, booleanDifference,
  isobands,
  embeddedIntersectionCurves, embeddedSelfIntersectionCurves,
  meshArrangements, polygonArrangements,
} from "./cut/sync";

// Intersect operations
export { intersectionCurves, selfIntersectionCurves, isocontours } from "./intersect/sync";
export type { IntersectOpts } from "./intersect/sync";
export type {
  LabeledCutResult, LabeledCutResultWithCurves,
  IsobandsResult, IsobandsResultWithCurves,
  CutResultWithCurves,
  MeshArrangementResult, MeshArrangementResultWithCurves,
  PolygonArrangementResult, PolygonArrangementResultWithCurves,
} from "./cut/sync";

// Spatial queries
export {
  distance, distance2, closestPoint, closestPointPair,
  neighborSearch, intersects, rayCast,
} from "./spatial/sync";
export type {
  Form,
  ClosestPointResult, ClosestPointBatchResult,
  ClosestPointPairResult, ClosestPointPairBatchResult,
  NeighborResult, NeighborBatchResult, NeighborPairResult,
  NeighborKnnResult, NeighborKnnBatchResult,
  KnnOptions, NeighborSearchOptions,
  RayCastResult, RayCastPrimBatchResult, RayCastFormBatchResult,
  RayCastOptions,
} from "./spatial/sync";

// Geometry operations
export {
  triangulate, sphereMesh, cylinderMesh, boxMesh, planeMesh, tubeMesh,
  area, signedVolume, volume, meanEdgeLength, minEdgeLength, maxEdgeLength,
  positivelyOriented,
  principalCurvatures, principalDirections, shapeIndex,
  laplacianSmoothed, taubinSmoothed,
  fitRigidAlignment, fitIcpAlignment, fitObbAlignment, chamferError,
} from "./geometry/sync";
export type { PrincipalCurvatures, PrincipalDirections, IcpOptions, ObbOptions, ChamferOptions } from "./geometry/sync";
export type { MeshLike } from "./form/MeshLike";

// Topology
export {
  isClosed, isOpen, isManifold, isNonManifold,
  eulerCharacteristic,
  boundaryEdges, nonManifoldEdges,
  boundaryPaths, kRings, neighborhoods, connectEdgesToPaths,
  labelConnectedComponents, connectedComponents,
  consistentlyOriented,
} from "./topology/sync";
export type { ConnectedComponentsResult, ComponentType } from "./topology/sync";

// Clean
export { cleaned } from "./clean/sync";
export type { CleanedMeshResult, CleanedPointsResult } from "./clean/sync";

// Reindex
export {
  reindexed, reindexedByMask, reindexedByIds,
  reindexedByMaskOnPoints, reindexedByIdsOnPoints,
  concatenateMeshes, splitIntoComponents,
} from "./reindex/sync";
export type { ReindexedMeshResult, SplitResult } from "./reindex/sync";

// Remesh operations
export { decimated, isotropicRemeshed } from "./remesh/sync";
export type { DecimateOptions, RemeshOptions } from "./remesh/sync";

// Async namespace
export * as async from "./async";

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

export {
  sum, min, max, mean, norm, argmin, argmax, any, all,
  sort, sort_, argsort, atan2,
  unique, setUnion, setIntersection, setDifference,
  bincount, histogram,
} from "../ndarray/async";
export { readStl, readObj, writeStl, writeObj } from "../io/async";
export {
  booleanUnion, booleanIntersection, booleanDifference,
  isobands,
  embeddedIntersectionCurves, embeddedSelfIntersectionCurves,
  meshArrangements, polygonArrangements,
} from "../cut/async";
export { intersectionCurves, selfIntersectionCurves, isocontours } from "../intersect/async";
export {
  distance, distance2, closestPoint, closestPointPair,
  neighborSearch, intersects, rayCast,
  buildTree,
} from "../spatial/async";
export {
  triangulate, sphereMesh, cylinderMesh, boxMesh, planeMesh, tubeMesh, sharpEdges,
  area, signedVolume, volume, meanEdgeLength, minEdgeLength, maxEdgeLength,
  reverseWinding, positivelyOriented,
  principalCurvatures, principalDirections, shapeIndex,
  laplacianSmoothed, taubinSmoothed,
  fitRigidAlignment, fitIcpAlignment, fitObbAlignment, chamferError,
  computeNormals, computePointNormals,
} from "../geometry/async";
export {
  isClosed, isOpen, isManifold, isNonManifold,
  eulerCharacteristic,
  boundaryEdges, nonManifoldEdges,
  boundaryPaths, kRings, neighborhoods, connectEdgesToPaths,
  labelConnectedComponents, connectedComponents,
  consistentlyOriented,
  computeFaceMembership, computeManifoldEdgeLink, computeFaceLink, computeVertexLink,
  cdt,
  domainLabels,
} from "../topology/async";
export { cleaned } from "../clean/async";
export {
  reindexed, reindexedByMask, reindexedByIds,
  reindexedByMaskOnPoints, reindexedByIdsOnPoints,
  concatenateMeshes, splitIntoComponents, splitIntoDomains,
} from "../reindex/async";
export { decimated, isotropicRemeshed } from "../remesh/async";
export { obbFrom } from "../primitive/async";

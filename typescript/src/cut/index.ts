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
  booleanUnion, booleanIntersection, booleanDifference,
  isobands,
  embeddedIntersectionCurves, embeddedSelfIntersectionCurves,
  meshArrangements,
} from "./sync";
export type {
  LabeledCutResult, LabeledCutResultWithCurves,
  IsobandsResult, IsobandsResultWithCurves,
  CutResultWithCurves,
  MeshArrangementResult, MeshArrangementResultWithCurves,
} from "./sync";
export * as async from "./async";

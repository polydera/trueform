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
  csgGraph, CsgGraph, Expr, op, outerShell,
  booleanUnion, booleanIntersection, booleanDifference,
} from "./sync";
export type {
  CsgGraphOptions, CsgSelectionOptions, CsgMeshOptions, CsgDomainsOptions,
  CsgMeshLabeledResult, CsgMeshIndexMapResult,
  CsgDomainsResult, CsgDomainsLabeledResult, CsgDomainsIndexMapResult,
  LabeledBooleanResult, LabeledBooleanResultWithCurves,
} from "./sync";
export * as async from "./async";

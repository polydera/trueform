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

import type {
  NDArrayBool, NDArrayFloat32, NDArrayFloat64, NDArrayInt32, NDArrayInt8,
} from "./NDArray";

/** All supported dtype string literals. */
export type Dtype = "float32" | "float64" | "int32" | "int8" | "bool";

/** Numeric dtypes (excludes bool, which is logically distinct). */
export type NumericDtype = "float32" | "float64" | "int32" | "int8";

/** Float dtypes only. */
export type FloatDtype = "float32" | "float64";

/** Dtypes supported by ordered/set operations (numeric, excludes bool). */
export type OrderedDtype = NumericDtype;

/** Dtypes supported by `random` (float + int32, no int8/bool). */
export type RandomDtype = "float32" | "float64" | "int32";

/** Dtypes supported by `eye` and `arange` (float + int32, no int8/bool). */
export type RangeDtype = "float32" | "float64" | "int32";

/** Dtypes supported by `full` (numeric only, no bool). */
export type FullDtype = NumericDtype;

/** Maps a Dtype string literal to its corresponding NDArray<T> alias. */
export type DtypeToArray<D extends Dtype> =
  D extends "float32" ? NDArrayFloat32 :
  D extends "float64" ? NDArrayFloat64 :
  D extends "int32"   ? NDArrayInt32 :
  D extends "int8"    ? NDArrayInt8 :
  D extends "bool"    ? NDArrayBool :
  never;

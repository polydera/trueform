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

export { NDArray, NDArrayInt8, NDArrayInt32, NDArrayFloat32, NDArrayBool } from "./NDArray";
export type { NativeNDArray } from "./NDArray";
export { OffsetBlockedBuffer } from "./OffsetBlockedBuffer";
export {
  ndarray, random, stack, concatenate, tile,
  where, zeros, ones, full, eye, arange, linspace,
  take, takeAlongAxis, sort, sort_, argsort,
  unique, setUnion, setIntersection, setDifference,
} from "./factories";
export { sum, min, max, mean, norm, argmin, argmax, any, all } from "./reductions";
export { bincount, histogram } from "./histogram";
export type { BincountOptions, HistogramOptions, HistogramResult } from "./histogram";
export {
  sqrt, sqrt_, sin, sin_, cos, cos_, tan, tan_,
  asin, asin_, acos, acos_, atan, atan_,
  exp, exp_, log, log_, log2, log2_, log10, log10_,
  floor, floor_, ceil, ceil_, round, round_,
  pow, pow_, abs, neg, clip,
  dot, cross, atan2,
  normalize, normalize_,
} from "./math";
export * as async from "./async";

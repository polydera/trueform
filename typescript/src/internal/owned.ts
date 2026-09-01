/*
 * Copyright (c) 2026 XLAB
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

import { NDArray } from "../ndarray/NDArray";

/**
 * An array a call creates only to feed native code is referenced by nobody
 * once its handle has been read, so the finalization registry may delete that
 * handle while a dispatched call still reads it. Every such array is tracked
 * here and released on the call's own terminal path instead.
 */

/** Track a wrapper the call created for the native side alone. */
export function own<T extends NDArray>(arr: T, owned: NDArray[]): T {
  owned.push(arr);
  return arr;
}

/** The array in `dtype`, converting and tracking it when it is not already. */
export function coerce(
  arr: NDArray, dtype: "float32" | "float64", owned: NDArray[],
): NDArray {
  return arr.dtype === dtype ? arr : own(arr.as(dtype), owned);
}

/** Release the tracked wrappers, newest first. */
export function disposeOwned(owned: NDArray[]): void {
  for (let i = owned.length; i-- > 0;) owned[i].delete();
}

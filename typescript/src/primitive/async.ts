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

import { native, dispatcher } from "../native";
import { NDArray } from "../ndarray/NDArray";
import { pointsOf, type OBB } from "./factories";
import { coerce, disposeOwned } from "../internal/owned";

/** Compute the oriented bounding box of points or a mesh (async). */
export async function obbFrom(input: NDArray): Promise<OBB>;
export async function obbFrom(input: { points: NDArray }): Promise<OBB>;
export async function obbFrom(
  input: NDArray | { points: NDArray },
): Promise<OBB> {
  const owned: NDArray[] = [];
  try {
    const pts = pointsOf(input, owned);
    const dt = pts.dtype === "float64" ? "float64" : "float32";
    const safe = coerce(pts, dt, owned);
    return await dispatcher().run(
      () => native()[`dispatch_obb_from_${dt}`](safe._handle),
      (raw) => ({
        origin: new NDArray(raw.origin, dt),
        axes: new NDArray(raw.axes, dt),
        extent: new NDArray(raw.extent, dt),
      }),
    );
  } finally {
    disposeOwned(owned);
  }
}

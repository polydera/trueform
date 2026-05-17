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
import type { OBB } from "./factories";

/** Compute the oriented bounding box of points or a mesh (async). */
export async function obbFrom(input: NDArray): Promise<OBB>;
export async function obbFrom(input: { points: NDArray }): Promise<OBB>;
export async function obbFrom(
  input: NDArray | { points: NDArray },
): Promise<OBB> {
  const pts = input instanceof NDArray ? input : input.points;
  const dt = pts.dtype === "float64" ? "float64" : "float32";
  const safe = pts.dtype === dt ? pts : pts.as(dt);
  return dispatcher().run(
    () => native()[`dispatch_obb_from_${dt}`](safe._handle),
    (raw) => ({
      origin: new NDArray(raw.origin, dt),
      axes: new NDArray(raw.axes, dt),
      extent: new NDArray(raw.extent, dt),
    }),
  );
}

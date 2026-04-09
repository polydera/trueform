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
import { NDArray, NDArrayFloat32 } from "../ndarray/NDArray";
import type { OBB } from "./factories";

/** Compute the oriented bounding box of points or a mesh (async). */
export async function obbFrom(input: NDArrayFloat32): Promise<OBB>;
export async function obbFrom(input: { points: NDArrayFloat32 }): Promise<OBB>;
export async function obbFrom(
  input: NDArrayFloat32 | { points: NDArrayFloat32 },
): Promise<OBB> {
  const pts = input instanceof NDArray ? input : input.points;
  const safe = pts.dtype === "float32" ? pts : pts.as("float32");
  return dispatcher().run(
    () => native().dispatch_obb_from(safe._handle),
    (raw) => ({
      origin: new NDArray(raw.origin, "float32"),
      axes: new NDArray(raw.axes, "float32"),
      extent: new NDArray(raw.extent, "float32"),
    }),
  );
}

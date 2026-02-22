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

import { native } from "../native";
import { NDArray, NDArrayInt32, NDArrayFloat32 } from "../ndarray/NDArray";
import { ndarray } from "../ndarray/factories";
import { Mesh } from "./Mesh";

/** Create a WASM-resident triangle mesh from face indices and point coordinates. */
export function mesh(faces: NDArrayInt32 | Int32Array, points: NDArrayFloat32 | Float32Array): Mesh {
  const f = faces instanceof NDArray ? faces : ndarray(faces, [faces.length / 3, 3]);
  const p = points instanceof NDArray ? points : ndarray(points, [points.length / 3, 3]);
  return new Mesh(native().NativeMesh.create(f._handle, p._handle));
}

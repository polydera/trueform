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
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";
import { ndarray } from "../ndarray/factories";
import { Mesh } from "./Mesh";
import { PointCloud } from "./PointCloud";
import { Curves } from "./Curves";

/** Create a WASM-resident triangle mesh from face indices and point coordinates. */
export function mesh(faces: NDArrayInt32 | Int32Array, points: NDArrayFloat32 | Float32Array): Mesh {
  const f = faces instanceof NDArray ? faces : ndarray(faces, [faces.length / 3, 3]);
  const p = points instanceof NDArray ? points : ndarray(points, [points.length / 3, 3]);
  return new Mesh(native().NativeMesh.create(f._handle, p._handle));
}

/** Create a WASM-resident point cloud from point coordinates or a Mesh. */
export function pointCloud(input: Mesh | NDArrayFloat32 | Float32Array): PointCloud {
  if (input instanceof Mesh) return PointCloud.fromMesh(input);
  const p = input instanceof NDArray ? input : ndarray(input, [input.length / 3, 3]);
  return new PointCloud(native().NativePointCloud.create(p._handle));
}

/** Create WASM-resident curves from path indices and point coordinates. */
export function curves(paths: OffsetBlockedBuffer, points: NDArrayFloat32 | Float32Array): Curves {
  const p = points instanceof NDArray ? points : ndarray(points, [points.length / 3, 3]);
  return new Curves(native().NativeCurves.create(paths._handle, p._handle));
}

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
import { NDArray, NDArrayInt32, NDArrayFloat32, NDArrayFloat64 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";
import { ndarray } from "../ndarray/factories";
import { Mesh } from "./Mesh";
import { PointCloud } from "./PointCloud";
import { Curves } from "./Curves";

/** Create a WASM-resident triangle mesh from face indices and point coordinates. */
export function mesh(
  faces: NDArrayInt32 | Int32Array,
  points: NDArrayFloat32 | NDArrayFloat64 | Float32Array | Float64Array,
): Mesh {
  const f = faces instanceof NDArray ? faces : ndarray(faces, [faces.length / 3, 3]);
  const p: NDArrayFloat32 | NDArrayFloat64 = points instanceof NDArray
    ? points
    : points instanceof Float64Array
      ? ndarray(points, [points.length / 3, 3])
      : ndarray(points, [points.length / 3, 3]);
  const dtype: "float32" | "float64" = p.dtype === "float64" ? "float64" : "float32";
  const m = native();
  const NativeMeshCtor = dtype === "float64" ? m.NativeFloat64Mesh : m.NativeFloat32Mesh;
  return new Mesh(NativeMeshCtor.create(f._handle, p._handle), dtype);
}

/** Create a WASM-resident point cloud from point coordinates or a Mesh. */
export function pointCloud(
  input: Mesh | NDArrayFloat32 | NDArrayFloat64 | Float32Array | Float64Array,
): PointCloud {
  if (input instanceof Mesh) return PointCloud.fromMesh(input);
  const p: NDArrayFloat32 | NDArrayFloat64 = input instanceof NDArray
    ? input
    : input instanceof Float64Array
      ? ndarray(input, [input.length / 3, 3])
      : ndarray(input, [input.length / 3, 3]);
  const dtype: "float32" | "float64" = p.dtype === "float64" ? "float64" : "float32";
  const n = native();
  const Native = dtype === "float64" ? n.NativeFloat64PointCloud : n.NativeFloat32PointCloud;
  return new PointCloud(Native.create(p._handle), dtype);
}

/** Create WASM-resident curves from path indices and point coordinates. */
export function curves(
  paths: OffsetBlockedBuffer,
  points: NDArrayFloat32 | NDArrayFloat64 | Float32Array | Float64Array,
): Curves {
  const p: NDArrayFloat32 | NDArrayFloat64 = points instanceof NDArray
    ? points
    : points instanceof Float64Array
      ? ndarray(points, [points.length / 3, 3])
      : ndarray(points, [points.length / 3, 3]);
  const dtype: "float32" | "float64" = p.dtype === "float64" ? "float64" : "float32";
  const n = native();
  const Native = dtype === "float64" ? n.NativeFloat64Curves : n.NativeFloat32Curves;
  return new Curves(Native.create(paths._handle, p._handle), dtype);
}

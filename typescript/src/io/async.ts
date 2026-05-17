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
import { Mesh } from "../form/Mesh";
import { NDArray, NDArrayInt8, NDArrayInt32, NDArrayFloat32, NDArrayFloat64 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";
import { triangulate } from "../geometry/async";
import type { ReadObjOptions } from "./sync";
import type { MeshLike } from "../form/MeshLike";

/** Read an STL file (binary or ASCII) into a triangle mesh, off the main thread.
 *  STL coordinates are float32 by format spec. */
export async function readStl(data: ArrayBuffer | Uint8Array): Promise<Mesh> {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  return dispatcher().run(
    () => native().dispatch_read_stl_buffer(bytes),
    (raw) => new Mesh(raw, "float32"),
  );
}

/** Read raw STL data as faces and points, off the main thread.
 *  STL coordinates are float32 by format spec. */
export async function readStlData(data: ArrayBuffer | Uint8Array): Promise<MeshLike> {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  return dispatcher().run(
    () => native().dispatch_read_stl_buffer(bytes),
    (raw): MeshLike => {
      const faces: NDArrayInt32 = new NDArray(raw.faces(), "int32");
      const points: NDArrayFloat32 = new NDArray(raw.points(), "float32");
      raw.delete();
      return { faces, points };
    },
  );
}

/** Read an OBJ file into a triangle mesh, off the main thread. */
export async function readObj(data: ArrayBuffer | Uint8Array, opts?: ReadObjOptions): Promise<Mesh> {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  const dtype = opts?.dtype ?? "float32";
  if (opts?.dynamic) {
    return triangulate(await readObjData(bytes, { dynamic: true, dtype }));
  }
  return dispatcher().run(
    () => native()[`dispatch_read_obj_buffer_${dtype}`](bytes),
    (raw) => new Mesh(raw, dtype),
  );
}

/** Read raw OBJ data as faces and points, off the main thread. */
export async function readObjData(data: ArrayBuffer | Uint8Array, opts?: ReadObjOptions): Promise<MeshLike> {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  const dtype = opts?.dtype ?? "float32";
  if (opts?.dynamic) {
    return dispatcher().run(
      () => native()[`dispatch_read_obj_buffer_data_${dtype}`](bytes),
      (raw): MeshLike => ({
        faces: new OffsetBlockedBuffer(raw.faces),
        points: new NDArray(raw.points, dtype) as NDArrayFloat32 | NDArrayFloat64,
      }),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_read_obj_buffer_${dtype}`](bytes),
    (raw): MeshLike => {
      const faces: NDArrayInt32 = new NDArray(raw.faces(), "int32");
      const points = new NDArray(raw.points(), dtype) as NDArrayFloat32 | NDArrayFloat64;
      raw.delete();
      return { faces, points };
    },
  );
}

/** Serialize a mesh to binary STL format, off the main thread.
 *  STL output is float32 by spec (float64 mesh coordinates are downcast). */
export async function writeStl(mesh: Mesh): Promise<NDArrayInt8> {
  return dispatcher().run(
    () => mesh.dtype === "float64"
      ? native().dispatch_write_stl_buffer_float64(mesh._handle)
      : native().dispatch_write_stl_buffer(mesh._handle),
    (raw) => new NDArray(raw, "int8"),
  );
}

/** Serialize a mesh to ASCII OBJ format, off the main thread. Precision follows
 *  the mesh dtype: float32 emits %.9g, float64 emits %.17g for exact round-trip. */
export async function writeObj(mesh: Mesh): Promise<NDArrayInt8> {
  return dispatcher().run(
    () => native()[`dispatch_write_obj_buffer_${mesh.dtype}`](mesh._handle),
    (raw) => new NDArray(raw, "int8"),
  );
}

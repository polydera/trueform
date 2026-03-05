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
import { Mesh } from "../form/Mesh";
import { NDArray, NDArrayInt8, NDArrayInt32, NDArrayFloat32 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";
import { triangulate } from "../geometry/sync";
import type { MeshLike } from "../form/MeshLike";

/** Read an STL file (binary or ASCII) into a triangle mesh. */
export function readStl(data: ArrayBuffer | Uint8Array): Mesh {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  return new Mesh(native().read_stl_buffer(bytes));
}

/** Read raw STL data as faces and points without creating a Mesh. */
export function readStlData(data: ArrayBuffer | Uint8Array): MeshLike {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  const raw = native().read_stl_buffer(bytes);
  const faces: NDArrayInt32 = new NDArray(raw.faces(), "int32");
  const points: NDArrayFloat32 = new NDArray(raw.points(), "float32");
  raw.delete();
  return { faces, points };
}

/** Options for reading OBJ files. */
export interface ReadObjOptions {
  /** Read all polygon sizes and auto-triangulate. Default: false (triangles only). */
  dynamic?: boolean;
}

/** Read an OBJ file into a triangle mesh. */
export function readObj(data: ArrayBuffer | Uint8Array, opts?: ReadObjOptions): Mesh {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  if (opts?.dynamic) {
    return triangulate(readObjData(bytes, { dynamic: true }));
  }
  return new Mesh(native().read_obj_buffer(bytes));
}

/** Read raw OBJ data as faces and points without creating a Mesh. */
export function readObjData(data: ArrayBuffer | Uint8Array, opts?: ReadObjOptions): MeshLike {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  if (opts?.dynamic) {
    const raw = native().read_obj_buffer_data(bytes);
    return {
      faces: new OffsetBlockedBuffer(raw.faces),
      points: new NDArray(raw.points, "float32"),
    };
  }
  const raw = native().read_obj_buffer(bytes);
  const faces: NDArrayInt32 = new NDArray(raw.faces(), "int32");
  const points: NDArrayFloat32 = new NDArray(raw.points(), "float32");
  raw.delete();
  return { faces, points };
}

/** Serialize a mesh to binary STL format. */
export function writeStl(mesh: Mesh): NDArrayInt8 {
  return new NDArray(native().write_stl_buffer(mesh._handle), "int8");
}

/** Serialize a mesh to ASCII OBJ format. */
export function writeObj(mesh: Mesh): NDArrayInt8 {
  return new NDArray(native().write_obj_buffer(mesh._handle), "int8");
}

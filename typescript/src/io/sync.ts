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
import { NDArray, NDArrayInt8 } from "../ndarray/NDArray";

/** Read an STL file (binary or ASCII) into a triangle mesh. */
export function readStl(data: ArrayBuffer | Uint8Array): Mesh {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  return new Mesh(native().read_stl_buffer(bytes));
}

/** Read an OBJ file (triangles only) into a triangle mesh. */
export function readObj(data: ArrayBuffer | Uint8Array): Mesh {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  return new Mesh(native().read_obj_buffer(bytes));
}

/** Serialize a mesh to binary STL format. */
export function writeStl(mesh: Mesh): NDArrayInt8 {
  return new NDArray(native().write_stl_buffer(mesh._handle), "int8");
}

/** Serialize a mesh to ASCII OBJ format. */
export function writeObj(mesh: Mesh): NDArrayInt8 {
  return new NDArray(native().write_obj_buffer(mesh._handle), "int8");
}

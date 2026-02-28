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
import { NDArray, NDArrayInt8 } from "../ndarray/NDArray";

/** Read an STL file (binary or ASCII) into a triangle mesh, off the main thread. */
export async function readStl(data: ArrayBuffer | Uint8Array): Promise<Mesh> {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  return dispatcher().run(
    () => native().dispatch_read_stl_buffer(bytes),
    (raw) => new Mesh(raw),
  );
}

/** Read an OBJ file (triangles only) into a triangle mesh, off the main thread. */
export async function readObj(data: ArrayBuffer | Uint8Array): Promise<Mesh> {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : data;
  return dispatcher().run(
    () => native().dispatch_read_obj_buffer(bytes),
    (raw) => new Mesh(raw),
  );
}

/** Serialize a mesh to binary STL format, off the main thread. */
export async function writeStl(mesh: Mesh): Promise<NDArrayInt8> {
  return dispatcher().run(
    () => native().dispatch_write_stl_buffer(mesh._handle),
    (raw) => new NDArray(raw, "int8"),
  );
}

/** Serialize a mesh to ASCII OBJ format, off the main thread. */
export async function writeObj(mesh: Mesh): Promise<NDArrayInt8> {
  return dispatcher().run(
    () => native().dispatch_write_obj_buffer(mesh._handle),
    (raw) => new NDArray(raw, "int8"),
  );
}

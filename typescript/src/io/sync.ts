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

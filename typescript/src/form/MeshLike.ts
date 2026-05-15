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

import { NDArrayInt32, NDArrayFloat32, NDArrayFloat64 } from "../ndarray/NDArray";
import { OffsetBlockedBuffer } from "../ndarray/OffsetBlockedBuffer";

/** Duck type for mesh-like objects with faces and points. */
export interface MeshLike {
  /** Face indices: fixed-size (NDArrayInt32 [F, N]) or variable-size (OffsetBlockedBuffer). */
  faces: NDArrayInt32 | OffsetBlockedBuffer;
  /** Vertex coordinates [V, 3]. */
  points: NDArrayFloat32 | NDArrayFloat64;
}

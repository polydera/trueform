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
import { Curves } from "../form/Curves";
import { NDArrayFloat32 } from "../ndarray/NDArray";

/** Compute intersection curves between two meshes. */
export function intersectionCurves(m0: Mesh, m1: Mesh): Curves {
  return new Curves(native().intersection_curves(m0._handle, m1._handle));
}

/** Find curves where a mesh intersects itself. */
export function selfIntersectionCurves(mesh: Mesh): Curves {
  return new Curves(native().self_intersection_curves(mesh._handle));
}

/** Extract isocontour curves from a scalar field at one or more thresholds. */
export function isocontours(
  mesh: Mesh, scalars: NDArrayFloat32, threshold: number | Float32Array,
): Curves {
  if (typeof threshold === "number") {
    return new Curves(
      native().isocontours(mesh._handle, scalars._handle, threshold),
    );
  }
  return new Curves(
    native().isocontours_multi(mesh._handle, scalars._handle, threshold),
  );
}

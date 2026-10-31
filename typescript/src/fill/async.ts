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
import type { FloatDtype } from "../ndarray/dtype";
import { Mesh } from "../form/Mesh";
import type { BoundaryRimsResult } from "../topology/sync";
import { fillArgs, filledArgs, wrapHoleFill } from "./sync";
import type { FillHolesOpts, HoleFill } from "./sync";

/** Fill a mesh's boundary rims, off the main thread. */
export async function fillHoles(
  mesh: Mesh, rims: BoundaryRimsResult, opts?: FillHolesOpts,
): Promise<HoleFill> {
  const dt = mesh.dtype as FloatDtype;
  const args = fillArgs(rims, opts);
  return dispatcher().run(
    () => native()[`dispatch_fill_holes_${dt}`](mesh._handle, ...args),
    (raw) => wrapHoleFill(raw, dt),
  );
}

/** Replay a fill into a mesh, off the main thread. */
export async function filledMesh(mesh: Mesh, fill: HoleFill): Promise<Mesh> {
  const dt = mesh.dtype as FloatDtype;
  const args = filledArgs(fill);
  return dispatcher().run(
    () => native()[`dispatch_filled_mesh_${dt}`](mesh._handle, ...args),
    (raw) => new Mesh(raw, dt),
  );
}

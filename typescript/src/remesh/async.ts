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

import type { DecimateOptions, RemeshOptions } from "./sync";

/** Decimate a mesh to a target proportion of original faces (async). Returns a new Mesh. */
export async function decimated(m: Mesh, targetProportion: number, opts?: DecimateOptions): Promise<Mesh> {
  return dispatcher().run(
    () => native().dispatch_decimated(
      m._handle,
      targetProportion,
      opts?.maxAspectRatio ?? 40,
      opts?.preserveBoundary ?? false,
      opts?.stabilizer ?? 1e-3,
      opts?.parallel ?? true,
    ),
    (raw) => new Mesh(raw),
  );
}

/** Isotropic remesh to uniform target edge length (async). Returns a new Mesh. */
export async function isotropicRemeshed(m: Mesh, targetLength: number, opts?: RemeshOptions): Promise<Mesh> {
  return dispatcher().run(
    () => native().dispatch_isotropic_remeshed(
      m._handle,
      targetLength,
      opts?.iterations ?? 3,
      opts?.relaxationIters ?? 3,
      opts?.maxAspectRatio ?? -1,
      opts?.lambda ?? 0.5,
      opts?.preserveBoundary ?? false,
      opts?.useQuadric ?? false,
      opts?.parallel ?? true,
    ),
    (raw) => new Mesh(raw),
  );
}

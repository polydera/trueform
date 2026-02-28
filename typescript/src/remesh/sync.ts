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

/** Options for quadric error metric decimation. */
export interface DecimateOptions {
  /** Maximum triangle aspect ratio after collapse. Negative to disable. Default: 40 */
  maxAspectRatio?: number;
  /** If true, boundary edges are never collapsed. Default: false */
  preserveBoundary?: boolean;
  /** Tikhonov stabilizer for quadric solve. Default: 1e-3 */
  stabilizer?: number;
  /** Use parallel partitioned collapse. Default: true */
  parallel?: boolean;
}

/** Options for isotropic remeshing. */
export interface RemeshOptions {
  /** Number of outer iterations (split + collapse + flip + relax). Default: 3 */
  iterations?: number;
  /** Tangential relaxation iterations per outer iteration. Default: 3 */
  relaxationIters?: number;
  /** Maximum aspect ratio after collapse. Negative to disable. Default: -1 */
  maxAspectRatio?: number;
  /** Damping factor for tangential relaxation in (0, 1]. Default: 0.5 */
  lambda?: number;
  /** If true, boundary edges are never split or collapsed. Default: false */
  preserveBoundary?: boolean;
  /** Use quadric error metric for collapse vertex placement. Default: false */
  useQuadric?: boolean;
  /** Use parallel execution. Default: true */
  parallel?: boolean;
}

/** Decimate a mesh to a target proportion of original faces. Returns a new Mesh. */
export function decimated(m: Mesh, targetProportion: number, opts?: DecimateOptions): Mesh {
  return new Mesh(native().decimated(
    m._handle,
    targetProportion,
    opts?.maxAspectRatio ?? 40,
    opts?.preserveBoundary ?? false,
    opts?.stabilizer ?? 1e-3,
    opts?.parallel ?? true,
  ));
}

/** Isotropic remesh to uniform target edge length. Returns a new Mesh. */
export function isotropicRemeshed(m: Mesh, targetLength: number, opts?: RemeshOptions): Mesh {
  return new Mesh(native().isotropic_remeshed(
    m._handle,
    targetLength,
    opts?.iterations ?? 3,
    opts?.relaxationIters ?? 3,
    opts?.maxAspectRatio ?? -1,
    opts?.lambda ?? 0.5,
    opts?.preserveBoundary ?? false,
    opts?.useQuadric ?? false,
    opts?.parallel ?? true,
  ));
}

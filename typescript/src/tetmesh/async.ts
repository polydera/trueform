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
import { tetMeshCall, wrapTetMesh } from "./sync";
import type { TetMeshOpts, TetMeshResult, TetMeshResultWithIndexMap } from "./sync";

/** Tetrahedralize the solid a closed triangle surface winds, off the main thread. */
export async function tetMesh(
  mesh: Mesh, opts: TetMeshOpts & { returnIndexMap: true },
): Promise<TetMeshResultWithIndexMap>;
export async function tetMesh(mesh: Mesh, opts?: TetMeshOpts): Promise<TetMeshResult>;
export async function tetMesh(
  mesh: Mesh, opts?: TetMeshOpts,
): Promise<TetMeshResult | TetMeshResultWithIndexMap> {
  const [name, args] = tetMeshCall(mesh, opts);
  return dispatcher().run(
    () => native()[`dispatch_${name}`](...args),
    (raw) => wrapTetMesh(raw, opts),
  );
}

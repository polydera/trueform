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
import { NDArray } from "../ndarray/NDArray";

import {
  _regionsNd,
  type DecimateOptions,
  type RemeshOptions,
  type SimplifyOptions,
  type RegionsInput,
  type RemeshWithRegions,
} from "./sync";

/** Decimate a mesh to a target proportion of original faces (async). Returns a new
 *  Mesh, or { mesh, regions } when preserveRegions is given. */
export function decimated(m: Mesh, targetProportion: number, opts?: DecimateOptions): Promise<Mesh>;
export function decimated(m: Mesh, targetProportion: number,
  opts: DecimateOptions & { preserveRegions: RegionsInput }): Promise<RemeshWithRegions>;
export async function decimated(m: Mesh, targetProportion: number,
  opts?: DecimateOptions): Promise<Mesh | RemeshWithRegions> {
  const fa = opts?.featureAngle != null && opts.featureAngle >= 0
    ? opts.featureAngle * Math.PI / 180 : -1;
  const dt = m.dtype;
  const regions = _regionsNd(opts?.preserveRegions, m.numberOfFaces);
  const wantRegions = opts?.preserveRegions != null;
  return dispatcher().run(
    () => native()[`dispatch_decimated_${dt}`](
      m._handle,
      targetProportion,
      opts?.minQuality ?? -1,
      opts?.preserveBoundary ?? true,
      opts?.stabilizer ?? 1e-3,
      opts?.parallel ?? true,
      fa,
      opts?.featureWeight ?? 100,
      regions._handle,
    ),
    (raw) => {
      const mesh = new Mesh(raw.mesh, dt);
      const outRegions = new NDArray<Int32Array>(raw.regions, "int32");
      return wantRegions ? { mesh, regions: outRegions } : mesh;
    },
  );
}

/** Simplify a mesh to an error budget using quadric error metrics (async). Returns a
 *  new Mesh, or { mesh, regions } when preserveRegions is given. */
export function simplified(m: Mesh, opts?: SimplifyOptions): Promise<Mesh>;
export function simplified(m: Mesh,
  opts: SimplifyOptions & { preserveRegions: RegionsInput }): Promise<RemeshWithRegions>;
export async function simplified(m: Mesh,
  opts?: SimplifyOptions): Promise<Mesh | RemeshWithRegions> {
  const fa = opts?.featureAngle != null && opts.featureAngle >= 0
    ? opts.featureAngle * Math.PI / 180 : -1;
  const dt = m.dtype;
  const regions = _regionsNd(opts?.preserveRegions, m.numberOfFaces);
  const wantRegions = opts?.preserveRegions != null;
  return dispatcher().run(
    () => native()[`dispatch_simplified_${dt}`](
      m._handle,
      opts?.errorRel ?? 0.002,
      opts?.optimizeIterations ?? 3,
      opts?.minQuality ?? 0.3,
      opts?.preserveBoundary ?? true,
      opts?.stabilizer ?? 1e-3,
      opts?.parallel ?? true,
      fa,
      opts?.featureWeight ?? 100,
      opts?.iterations ?? 1,
      opts?.relaxationIters ?? 3,
      opts?.lambda ?? 0.5,
      regions._handle,
    ),
    (raw) => {
      const mesh = new Mesh(raw.mesh, dt);
      const outRegions = new NDArray<Int32Array>(raw.regions, "int32");
      return wantRegions ? { mesh, regions: outRegions } : mesh;
    },
  );
}

/** Isotropic remesh to uniform target edge length (async). Returns a new Mesh, or
 *  { mesh, regions } when preserveRegions is given. */
export function isotropicRemeshed(m: Mesh, targetLength: number, opts?: RemeshOptions): Promise<Mesh>;
export function isotropicRemeshed(m: Mesh, targetLength: number,
  opts: RemeshOptions & { preserveRegions: RegionsInput }): Promise<RemeshWithRegions>;
export async function isotropicRemeshed(m: Mesh, targetLength: number,
  opts?: RemeshOptions): Promise<Mesh | RemeshWithRegions> {
  const fa = opts?.featureAngle != null && opts.featureAngle >= 0
    ? opts.featureAngle * Math.PI / 180 : -1;
  const dt = m.dtype;
  const regions = _regionsNd(opts?.preserveRegions, m.numberOfFaces);
  const wantRegions = opts?.preserveRegions != null;
  return dispatcher().run(
    () => native()[`dispatch_isotropic_remeshed_${dt}`](
      m._handle,
      targetLength,
      opts?.iterations ?? 3,
      opts?.relaxationIters ?? 3,
      opts?.minQuality ?? 0.3,
      opts?.lambda ?? 0.5,
      opts?.preserveBoundary ?? true,
      opts?.useQuadric ?? false,
      opts?.parallel ?? true,
      fa,
      opts?.featureWeight ?? 100,
      regions._handle,
    ),
    (raw) => {
      const mesh = new Mesh(raw.mesh, dt);
      const outRegions = new NDArray<Int32Array>(raw.regions, "int32");
      return wantRegions ? { mesh, regions: outRegions } : mesh;
    },
  );
}

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
import { NDArray, NDArrayBool, NDArrayFloat32, NDArrayFloat64, NDArrayInt32 } from "../ndarray/NDArray";
import type { FloatDtype } from "../ndarray/dtype";
import { Mesh } from "../form/Mesh";
import { Curves } from "../form/Curves";
import { assertSameDtype } from "../internal/dtype";
import { registry } from "../internal/registry";
import { Expr, programOf } from "./expr";

// ============================================================================
// Options / results
// ============================================================================

export interface CsgGraphOptions {
  /** Operand indices declared as oriented open sheets. */
  sheets?: number[];
  /** Intersection mode: "sos" or "primitives" (default). */
  mode?: "sos" | "primitives";
  /** World-coordinate predicate tolerance band (0 = exact). */
  tolerance?: number;
  /** Resolve crossings between contours on one face (default true). */
  resolveCrossings?: boolean;
  /**
   * Also intersect each operand with itself — required when an operand
   * can self-overlap, e.g. meshes concatenated into one operand. Domain
   * extraction then classifies the overlap pockets structurally; boolean
   * expressions still require solid, non-self-overlapping operands.
   */
  within?: boolean;
  /**
   * Cut-surface triangulation: "cdt" (plain constrained Delaunay per cut
   * loop, default) or "refinedCdt" (quality refinement of the cut surface;
   * shared boundaries stay watertight by construction).
   */
  triangulation?: "cdt" | "refinedCdt";
}

export interface CsgDomainsOptions {
  /** Drop the unbounded outside domain (default true). */
  excludeOuterShell?: boolean;
  /** Fuse open fragments instead of letting them partition (default true). */
  ignoreOpenFragments?: boolean;
}

export interface CsgMeshLabeledResult {
  mesh: Mesh;
  /** Per output face: the input form it came from. */
  tagLabels: NDArrayInt32;
  /** Per output face: the original face id within that form. */
  faceLabels: NDArrayInt32;
}

export interface CsgMeshIndexMapResult {
  mesh: Mesh;
  /** Output point -> input mesh tag; created points carry `nTags`. */
  pointTagLabels: NDArrayInt32;
  /** Output point -> input point id; created points carry `nOutputPoints`. */
  pointLabels: NDArrayInt32;
  /** Output face -> input mesh tag. */
  faceTagLabels: NDArrayInt32;
  /** Output face -> input face id (origin face for cut faces). */
  faceLabels: NDArrayInt32;
  /** Forward point map blocks: pointFOffsets[tag] slices pointFData. */
  pointFOffsets: NDArrayInt32;
  pointFData: NDArrayInt32;
  nOriginalPoints: number;
  nOriginalFaces: number;
  nTags: number;
  nOutputPoints: number;
}

export interface CsgDomainsResult {
  /** One watertight mesh per kept domain. */
  meshes: Mesh[];
  /** ids[k] = coarse domain id of cell k. */
  ids: NDArrayInt32;
}

export interface CsgDomainsLabeledResult extends CsgDomainsResult {
  /** Per-cell face -> input form blocks (offsets parallel to `meshes`). */
  tagOffsets: NDArrayInt32;
  tagData: NDArrayInt32;
  /** Per-cell face -> original face id blocks. */
  faceOffsets: NDArrayInt32;
  faceData: NDArrayInt32;
}

export interface CsgDomainsIndexMapResult extends CsgDomainsResult {
  faceTagOffsets: NDArrayInt32;
  faceTagData: NDArrayInt32;
  faceOffsets: NDArrayInt32;
  faceData: NDArrayInt32;
  pointTagOffsets: NDArrayInt32;
  pointTagData: NDArrayInt32;
  pointOffsets: NDArrayInt32;
  pointData: NDArrayInt32;
  nOriginalPoints: number;
  nTags: number;
  nOutputPoints: number;
  /** `[nCells, nTags]` boolean matrix: cell k inside form i (sheet:
   *  behind its normal). Row-major; mask it to select cells
   *  post-extraction. */
  inclusion: NDArrayBool;
}

// ============================================================================
// Internal helpers (shared with async.ts)
// ============================================================================

const MODE_MAP = { sos: 1, primitives: 2 } as const;
const RESOLVE_CROSSINGS = 4;
const WITHIN = 24; // self_intersections | resolve_self_crossing_contours
const TRIANGULATION_MAP = { cdt: 0, refinedCdt: 1 } as const;
const EXCLUDE_OUTER_SHELL = 1;
const IGNORE_OPEN_FRAGMENTS = 2;

/** @internal */
export function buildCsgConfig(meshes: Mesh[], opts?: CsgGraphOptions) {
  if (meshes.length < 2) throw new Error("csgGraph: need at least 2 meshes");
  assertSameDtype(meshes, meshes.map((_, i) => `mesh${i}`));
  const sheets = opts?.sheets ?? [];
  for (const s of sheets) {
    if (!Number.isInteger(s) || s < 0 || s >= meshes.length) {
      throw new RangeError(`sheet index ${s} out of range`);
    }
  }
  let mode: number = MODE_MAP[opts?.mode ?? "primitives"];
  if (opts?.resolveCrossings ?? true) mode |= RESOLVE_CROSSINGS;
  if (opts?.within) mode |= WITHIN;
  return {
    sheets,
    mode,
    tolerance: opts?.tolerance ?? 0,
    triangulation: TRIANGULATION_MAP[opts?.triangulation ?? "cdt"],
  };
}

/** @internal */
export function buildDomainsConfig(opts?: CsgDomainsOptions): number {
  let cfg = 0;
  if (opts?.excludeOuterShell ?? true) cfg |= EXCLUDE_OUTER_SHELL;
  if (opts?.ignoreOpenFragments ?? true) cfg |= IGNORE_OPEN_FRAGMENTS;
  return cfg;
}

/** @internal — `(expr?, opts?)` where the expr slot may hold the
 * options object instead (the clean/sync.ts discrimination idiom). */
export function splitExprArgs(
  exprOrOpts?: Expr | number | object,
  maybeOpts?: object,
): [Expr | number | undefined, any] {
  const isExpr =
    exprOrOpts instanceof Expr || typeof exprOrOpts === "number";
  if (isExpr) return [exprOrOpts as Expr | number, maybeOpts];
  if (maybeOpts !== undefined) {
    throw new TypeError("the expression must come before the options object");
  }
  return [undefined, exprOrOpts];
}

/** @internal */
export function wrapCsgMeshLabeled(raw: any, dt: FloatDtype): CsgMeshLabeledResult {
  return {
    mesh: new Mesh(raw.mesh, dt),
    tagLabels: new NDArray(raw.tagLabels, "int32"),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
  };
}

/** @internal */
export function wrapCsgMeshIndexMap(raw: any, dt: FloatDtype): CsgMeshIndexMapResult {
  return {
    mesh: new Mesh(raw.mesh, dt),
    pointTagLabels: new NDArray(raw.pointTagLabels, "int32"),
    pointLabels: new NDArray(raw.pointLabels, "int32"),
    faceTagLabels: new NDArray(raw.faceTagLabels, "int32"),
    faceLabels: new NDArray(raw.faceLabels, "int32"),
    pointFOffsets: new NDArray(raw.pointFOffsets, "int32"),
    pointFData: new NDArray(raw.pointFData, "int32"),
    nOriginalPoints: raw.nOriginalPoints,
    nOriginalFaces: raw.nOriginalFaces,
    nTags: raw.nTags,
    nOutputPoints: raw.nOutputPoints,
  };
}

function wrapMeshVector(raw: any, dt: FloatDtype): Mesh[] {
  const out: Mesh[] = [];
  const n = raw.size();
  for (let i = 0; i < n; ++i) out.push(new Mesh(raw.get(i), dt));
  raw.delete();
  return out;
}

/** @internal */
export function wrapCsgDomains(raw: any, dt: FloatDtype): CsgDomainsResult {
  return {
    meshes: wrapMeshVector(raw.meshes, dt),
    ids: new NDArray(raw.ids, "int32"),
  };
}

/** @internal */
export function wrapCsgDomainsLabeled(raw: any, dt: FloatDtype): CsgDomainsLabeledResult {
  return {
    meshes: wrapMeshVector(raw.meshes, dt),
    ids: new NDArray(raw.ids, "int32"),
    tagOffsets: new NDArray(raw.tagOffsets, "int32"),
    tagData: new NDArray(raw.tagData, "int32"),
    faceOffsets: new NDArray(raw.faceOffsets, "int32"),
    faceData: new NDArray(raw.faceData, "int32"),
  };
}

/** @internal */
export function wrapCsgDomainsIndexMap(raw: any, dt: FloatDtype): CsgDomainsIndexMapResult {
  return {
    meshes: wrapMeshVector(raw.meshes, dt),
    ids: new NDArray(raw.ids, "int32"),
    faceTagOffsets: new NDArray(raw.faceTagOffsets, "int32"),
    faceTagData: new NDArray(raw.faceTagData, "int32"),
    faceOffsets: new NDArray(raw.faceOffsets, "int32"),
    faceData: new NDArray(raw.faceData, "int32"),
    pointTagOffsets: new NDArray(raw.pointTagOffsets, "int32"),
    pointTagData: new NDArray(raw.pointTagData, "int32"),
    pointOffsets: new NDArray(raw.pointOffsets, "int32"),
    pointData: new NDArray(raw.pointData, "int32"),
    nOriginalPoints: raw.nOriginalPoints,
    nTags: raw.nTags,
    nOutputPoints: raw.nOutputPoints,
    inclusion: new NDArray(raw.inclusion, "bool"),
  };
}

// ============================================================================
// CsgGraph
// ============================================================================

/**
 * The arrangement of N meshes with its domain classification, built once;
 * any boolean expression over the operands is then a cheap query against
 * the same build.
 *
 * Holds the input meshes (`.forms`), the declared `.sheets`, and the
 * construction options; the native engine underneath is opaque. Free
 * explicitly with `delete()` (or `using`); a FinalizationRegistry backstop
 * releases leaked handles.
 *
 * @example
 * const graph = csgGraph([a, b, c], { triangulation: "refinedCdt" });
 * const diff = graph.mesh(op(0).sub(op(1)));
 * const { meshes, ids } = graph.domains();
 * graph.delete();
 */
export class CsgGraph {
  /** @internal */
  readonly _handle: any;
  /** Runtime dtype discriminator, mirrors the input meshes. */
  readonly dtype: FloatDtype;
  /** The input meshes, as passed. */
  readonly forms: Mesh[];
  /** Operand indices declared as sheets, as passed. */
  readonly sheets: number[];
  /** Construction options, echoed back. */
  readonly config: {
    mode: "sos" | "primitives";
    tolerance: number;
    resolveCrossings: boolean;
    triangulation: "cdt" | "refinedCdt";
  };

  private _createdPoints: NDArrayFloat32 | NDArrayFloat64 | null = null;

  /** @internal */
  constructor(
    handle: any,
    dtype: FloatDtype,
    forms: Mesh[],
    sheets: number[],
    config: CsgGraph["config"],
  ) {
    this._handle = handle;
    this.dtype = dtype;
    this.forms = forms;
    this.sheets = sheets;
    this.config = config;
    registry.register(this, { handle });
  }

  /**
   * Points the arrangement created (intersections; plus refinement points
   * under `triangulation: "refinedCdt"`), NDArray [K, 3] in the input dtype.
   */
  get createdPoints(): NDArrayFloat32 | NDArrayFloat64 {
    if (this._createdPoints === null) {
      this._createdPoints = new NDArray(
        native()[`csg_created_points_${this.dtype}`](this._handle),
        this.dtype,
      ) as NDArrayFloat32 | NDArrayFloat64;
    }
    return this._createdPoints;
  }

  /**
   * The boolean result mesh for `expr`; with no expression, the full
   * arrangement mesh (every input face, cut at intersections).
   * `returnIndexMap` requires an expression.
   */
  mesh(): Mesh;
  mesh(expr: Expr | number): Mesh;
  mesh(opts: { returnSourceIds: true }): CsgMeshLabeledResult;
  mesh(expr: Expr | number, opts: { returnSourceIds: true }): CsgMeshLabeledResult;
  mesh(expr: Expr | number, opts: { returnIndexMap: true }): CsgMeshIndexMapResult;
  mesh(
    exprOrOpts?: Expr | number | { returnSourceIds?: true; returnIndexMap?: true },
    maybeOpts?: { returnSourceIds?: true; returnIndexMap?: true },
  ): Mesh | CsgMeshLabeledResult | CsgMeshIndexMapResult {
    const [expr, opts] = splitExprArgs(exprOrOpts, maybeOpts);
    const program = expr === undefined ? [] : programOf(expr);
    const dt = this.dtype;
    if (opts?.returnSourceIds && opts?.returnIndexMap) {
      throw new Error("returnSourceIds and returnIndexMap are exclusive");
    }
    if (opts?.returnIndexMap) {
      return wrapCsgMeshIndexMap(
        native()[`csg_mesh_with_index_map_${dt}`](this._handle, program), dt,
      );
    }
    if (opts?.returnSourceIds) {
      return wrapCsgMeshLabeled(
        native()[`csg_mesh_with_labels_${dt}`](this._handle, program), dt,
      );
    }
    return new Mesh(native()[`csg_mesh_${dt}`](this._handle, program), dt);
  }

  /**
   * The intersection polylines of the arrangement — where two operand
   * surfaces cross (coincident walls excluded).
   */
  intersectionCurves(): Curves {
    return new Curves(
      native()[`csg_intersection_curves_${this.dtype}`](this._handle),
      this.dtype,
    );
  }

  /**
   * Every kept volumetric domain as its own watertight mesh; with an
   * expression, only domains inside its selection. The expression may be
   * omitted entirely: `domains()`, `domains(opts)`, `domains(expr)`,
   * `domains(expr, opts)` are all accepted.
   */
  domains(): CsgDomainsResult;
  domains(expr: Expr | number): CsgDomainsResult;
  domains(opts: CsgDomainsOptions & { returnSourceIds: true }): CsgDomainsLabeledResult;
  domains(opts: CsgDomainsOptions & { returnIndexMap: true }): CsgDomainsIndexMapResult;
  domains(opts: CsgDomainsOptions): CsgDomainsResult;
  domains(expr: Expr | number, opts: CsgDomainsOptions & { returnSourceIds: true }): CsgDomainsLabeledResult;
  domains(expr: Expr | number, opts: CsgDomainsOptions & { returnIndexMap: true }): CsgDomainsIndexMapResult;
  domains(expr: Expr | number, opts: CsgDomainsOptions): CsgDomainsResult;
  domains(
    exprOrOpts?: Expr | number | (CsgDomainsOptions & { returnSourceIds?: true; returnIndexMap?: true }),
    maybeOpts?: CsgDomainsOptions & { returnSourceIds?: true; returnIndexMap?: true },
  ): CsgDomainsResult | CsgDomainsLabeledResult | CsgDomainsIndexMapResult {
    const [expr, opts] = splitExprArgs(exprOrOpts, maybeOpts);
    const program = expr === undefined ? [] : programOf(expr);
    const cfg = buildDomainsConfig(opts);
    const dt = this.dtype;
    if (opts?.returnSourceIds && opts?.returnIndexMap) {
      throw new Error("returnSourceIds and returnIndexMap are exclusive");
    }
    if (opts?.returnIndexMap) {
      return wrapCsgDomainsIndexMap(
        native()[`csg_domains_with_index_map_${dt}`](this._handle, program, cfg), dt,
      );
    }
    if (opts?.returnSourceIds) {
      return wrapCsgDomainsLabeled(
        native()[`csg_domains_with_labels_${dt}`](this._handle, program, cfg), dt,
      );
    }
    return wrapCsgDomains(
      native()[`csg_domains_${dt}`](this._handle, program, cfg), dt,
    );
  }

  delete(): void {
    this._handle.destroy();
  }

  [Symbol.dispose](): void {
    this.delete();
  }
}

// ============================================================================
// Outer shell
// ============================================================================

/** Options for {@link outerShell}. */
export interface OuterShellOptions {
  /**
   * Mask open fragments (surface pieces carrying boundary edges) out of
   * the region formation instead of letting them partition (default false).
   */
  ignoreOpenFragments?: boolean;
}

/** @internal — also used by the async wrapper. */
export function buildOuterShellConfig(opts?: OuterShellOptions): number {
  return opts?.ignoreOpenFragments ? IGNORE_OPEN_FRAGMENTS : 0;
}

/**
 * Repair a mesh to its outer shell: the boundary of the union of
 * everything it encloses.
 *
 * Splits the mesh at its self-intersection curves, labels the volumetric
 * domains, and keeps only the faces bounding the unbounded outside,
 * oriented outward. Internal structure — overlap membranes between
 * interpenetrating parts, faces buried inside the solid, enclosed
 * cavities — is removed. The result is free of self-intersections and
 * suitable as a boolean or csg-graph operand.
 */
export function outerShell(mesh: Mesh, opts?: OuterShellOptions): Mesh {
  const dt = mesh.dtype;
  const cfg = buildOuterShellConfig(opts);
  return new Mesh(native()[`outer_shell_${dt}`](mesh._handle, cfg), dt);
}

// ============================================================================
// Factory
// ============================================================================

/**
 * Build the CSG graph of `meshes` once; query it arbitrarily often with
 * `graph.mesh(expr)` and `graph.domains()`.
 *
 * @param meshes - Two or more closed 3D triangle meshes, same dtype.
 * @param opts - Sheets, intersection mode/tolerance, cut-surface triangulation.
 */
export function csgGraph(meshes: Mesh[], opts?: CsgGraphOptions): CsgGraph {
  const cfg = buildCsgConfig(meshes, opts);
  const dt = meshes[0].dtype;
  const handle = native()[`csg_graph_${dt}`](
    meshes.map((m) => m._handle),
    cfg.sheets,
    cfg.mode,
    cfg.tolerance,
    cfg.triangulation,
  );
  return new CsgGraph(handle, dt, [...meshes], [...cfg.sheets], {
    mode: opts?.mode ?? "primitives",
    tolerance: cfg.tolerance,
    resolveCrossings: opts?.resolveCrossings ?? true,
    triangulation: opts?.triangulation ?? "cdt",
  });
}

export { Expr, op } from "./expr";

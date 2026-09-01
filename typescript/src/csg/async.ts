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
import { Curves } from "../form/Curves";
import { assertSameDtype } from "../internal/dtype";
import {
  CsgGraph,
  CsgGraphOptions,
  CsgDomainsOptions,
  CsgMeshOptions,
  CsgMeshLabeledResult,
  CsgMeshIndexMapResult,
  CsgDomainsResult,
  CsgDomainsLabeledResult,
  CsgDomainsIndexMapResult,
  LabeledBooleanResult,
  LabeledBooleanResultWithCurves,
  buildCsgConfig,
  buildDomainsConfig,
  meshRestriction,
  selectionTags,
  splitExprArgs,
  toSheets,
  wrapCsgMeshLabeled,
  wrapCsgMeshIndexMap,
  wrapCsgDomains,
  wrapCsgDomainsLabeled,
  wrapCsgDomainsIndexMap,
  wrapLabeledBoolean,
  wrapLabeledBooleanWithCurves,
} from "./sync";
import { Expr, programOf } from "./expr";

/**
 * Build the CSG graph on a worker; the returned graph is the same
 * stateful object `csgGraph` builds synchronously.
 */
export async function csgGraph(
  meshes: Mesh[], opts?: CsgGraphOptions,
): Promise<CsgGraph> {
  const cfg = buildCsgConfig(meshes, opts);
  const dt = meshes[0].dtype;
  return dispatcher().run(
    () => native()[`dispatch_csg_graph_${dt}`](
      meshes.map((m) => m._handle),
      cfg.sheets,
      cfg.mode,
      cfg.tolerance,
      cfg.triangulation,
    ),
    (raw) => new CsgGraph(raw, dt, [...meshes], [...cfg.sheets], {
      mode: opts?.mode ?? "primitives",
      tolerance: cfg.tolerance,
      resolveCrossings: opts?.resolveCrossings ?? true,
      triangulation: opts?.triangulation ?? "cdt",
    }),
  );
}

/** The boolean result mesh for `expr`, restricted to `selection`'s
 * surfaces or to the part of `inside`'s surfaces that lies within the
 * expression's region, evaluated on a worker; with none of them, the
 * full arrangement mesh. `returnIndexMap` requires an expression or a
 * selection. */
export async function csgMesh(graph: CsgGraph): Promise<Mesh>;
export async function csgMesh(graph: CsgGraph, expr: Expr | number): Promise<Mesh>;
export async function csgMesh(
  graph: CsgGraph, opts: CsgMeshOptions & { returnSourceIds: true },
): Promise<CsgMeshLabeledResult>;
export async function csgMesh(
  graph: CsgGraph, opts: { selection: number[]; returnIndexMap: true },
): Promise<CsgMeshIndexMapResult>;
export async function csgMesh(
  graph: CsgGraph, opts: CsgMeshOptions,
): Promise<Mesh>;
export async function csgMesh(
  graph: CsgGraph, expr: Expr | number,
  opts: CsgMeshOptions & { returnSourceIds: true },
): Promise<CsgMeshLabeledResult>;
export async function csgMesh(
  graph: CsgGraph, expr: Expr | number,
  opts: CsgMeshOptions & { returnIndexMap: true },
): Promise<CsgMeshIndexMapResult>;
export async function csgMesh(
  graph: CsgGraph, expr: Expr | number, opts: CsgMeshOptions,
): Promise<Mesh>;
export async function csgMesh(
  graph: CsgGraph,
  exprOrOpts?: Expr | number | (CsgMeshOptions & { returnSourceIds?: true; returnIndexMap?: true }),
  maybeOpts?: CsgMeshOptions & { returnSourceIds?: true; returnIndexMap?: true },
): Promise<Mesh | CsgMeshLabeledResult | CsgMeshIndexMapResult> {
  const [expr, opts] = splitExprArgs(exprOrOpts, maybeOpts);
  const program = expr === undefined ? [] : programOf(expr);
  const [tags, kind] = meshRestriction(graph, expr !== undefined, opts);
  const dt = graph.dtype;
  if (opts?.returnSourceIds && opts?.returnIndexMap) {
    throw new Error("returnSourceIds and returnIndexMap are exclusive");
  }
  if (opts?.returnIndexMap) {
    return dispatcher().run(
      () => native()[`dispatch_csg_mesh_with_index_map_${dt}`](graph._handle, program, tags, kind),
      (raw) => wrapCsgMeshIndexMap(raw, dt),
    );
  }
  if (opts?.returnSourceIds) {
    return dispatcher().run(
      () => native()[`dispatch_csg_mesh_with_labels_${dt}`](graph._handle, program, tags, kind),
      (raw) => wrapCsgMeshLabeled(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_csg_mesh_${dt}`](graph._handle, program, tags, kind),
    (raw) => new Mesh(raw, dt),
  );
}

/** Intersection polylines of the arrangement, extracted on a worker. */
export async function csgIntersectionCurves(graph: CsgGraph): Promise<Curves> {
  const dt = graph.dtype;
  return dispatcher().run(
    () => native()[`dispatch_csg_intersection_curves_${dt}`](graph._handle),
    (raw) => new Curves(raw, dt),
  );
}

/** Kept volumetric domains, extracted on a worker; a `selection` leaves
 * each cell only the named forms' walls. The expression may be omitted:
 * `csgDomains(g)`, `csgDomains(g, opts)`, `csgDomains(g, expr)`,
 * `csgDomains(g, expr, opts)` are all accepted. */
export async function csgDomains(graph: CsgGraph): Promise<CsgDomainsResult>;
export async function csgDomains(
  graph: CsgGraph, expr: Expr | number,
): Promise<CsgDomainsResult>;
export async function csgDomains(
  graph: CsgGraph, opts: CsgDomainsOptions & { returnSourceIds: true },
): Promise<CsgDomainsLabeledResult>;
export async function csgDomains(
  graph: CsgGraph, opts: CsgDomainsOptions & { returnIndexMap: true },
): Promise<CsgDomainsIndexMapResult>;
export async function csgDomains(
  graph: CsgGraph, opts: CsgDomainsOptions,
): Promise<CsgDomainsResult>;
export async function csgDomains(
  graph: CsgGraph, expr: Expr | number,
  opts: CsgDomainsOptions & { returnSourceIds: true },
): Promise<CsgDomainsLabeledResult>;
export async function csgDomains(
  graph: CsgGraph, expr: Expr | number,
  opts: CsgDomainsOptions & { returnIndexMap: true },
): Promise<CsgDomainsIndexMapResult>;
export async function csgDomains(
  graph: CsgGraph, expr: Expr | number, opts: CsgDomainsOptions,
): Promise<CsgDomainsResult>;
export async function csgDomains(
  graph: CsgGraph,
  exprOrOpts?: Expr | number | (CsgDomainsOptions & { returnSourceIds?: true; returnIndexMap?: true }),
  maybeOpts?: CsgDomainsOptions & { returnSourceIds?: true; returnIndexMap?: true },
): Promise<CsgDomainsResult | CsgDomainsLabeledResult | CsgDomainsIndexMapResult> {
  const [expr, opts] = splitExprArgs(exprOrOpts, maybeOpts);
  const program = expr === undefined ? [] : programOf(expr);
  const tags = selectionTags(graph, opts);
  const cfg = buildDomainsConfig(opts);
  const dt = graph.dtype;
  if (opts?.returnSourceIds && opts?.returnIndexMap) {
    throw new Error("returnSourceIds and returnIndexMap are exclusive");
  }
  if (opts?.returnIndexMap) {
    return dispatcher().run(
      () => native()[`dispatch_csg_domains_with_index_map_${dt}`](graph._handle, program, tags, cfg),
      (raw) => wrapCsgDomainsIndexMap(raw, dt),
    );
  }
  if (opts?.returnSourceIds) {
    return dispatcher().run(
      () => native()[`dispatch_csg_domains_with_labels_${dt}`](graph._handle, program, tags, cfg),
      (raw) => wrapCsgDomainsLabeled(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_csg_domains_${dt}`](graph._handle, program, tags, cfg),
    (raw) => wrapCsgDomains(raw, dt),
  );
}

/**
 * Repair a mesh to its outer shell on a worker: the boundary of the union
 * of everything it encloses. Same semantics as the sync `outerShell`.
 */
export async function outerShell(mesh: Mesh): Promise<Mesh> {
  const dt = mesh.dtype;
  return dispatcher().run(
    () => native()[`dispatch_outer_shell_${dt}`](mesh._handle),
    (raw) => new Mesh(raw, dt),
  );
}

// ============================================================================
// Booleans
// ============================================================================

export async function booleanUnion(
  m0: Mesh, m1: Mesh, opts: { returnCurves: true; sheets?: number[] },
): Promise<LabeledBooleanResultWithCurves>;
export async function booleanUnion(
  m0: Mesh, m1: Mesh, opts?: { sheets?: number[] },
): Promise<LabeledBooleanResult>;
export async function booleanUnion(
  m0: Mesh, m1: Mesh, opts?: { returnCurves?: true; sheets?: number[] },
): Promise<LabeledBooleanResult | LabeledBooleanResultWithCurves> {
  assertSameDtype([m0, m1], ["mesh0", "mesh1"]);
  const dt = m0.dtype;
  const sheets = toSheets(opts?.sheets);
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native()[`dispatch_boolean_union_with_curves_${dt}`](m0._handle, m1._handle, sheets),
      (raw) => wrapLabeledBooleanWithCurves(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_boolean_union_${dt}`](m0._handle, m1._handle, sheets),
    (raw) => wrapLabeledBoolean(raw, dt),
  );
}

export async function booleanIntersection(
  m0: Mesh, m1: Mesh, opts: { returnCurves: true; sheets?: number[] },
): Promise<LabeledBooleanResultWithCurves>;
export async function booleanIntersection(
  m0: Mesh, m1: Mesh, opts?: { sheets?: number[] },
): Promise<LabeledBooleanResult>;
export async function booleanIntersection(
  m0: Mesh, m1: Mesh, opts?: { returnCurves?: true; sheets?: number[] },
): Promise<LabeledBooleanResult | LabeledBooleanResultWithCurves> {
  assertSameDtype([m0, m1], ["mesh0", "mesh1"]);
  const dt = m0.dtype;
  const sheets = toSheets(opts?.sheets);
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native()[`dispatch_boolean_intersection_with_curves_${dt}`](m0._handle, m1._handle, sheets),
      (raw) => wrapLabeledBooleanWithCurves(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_boolean_intersection_${dt}`](m0._handle, m1._handle, sheets),
    (raw) => wrapLabeledBoolean(raw, dt),
  );
}

export async function booleanDifference(
  m0: Mesh, m1: Mesh, opts: { returnCurves: true; sheets?: number[] },
): Promise<LabeledBooleanResultWithCurves>;
export async function booleanDifference(
  m0: Mesh, m1: Mesh, opts?: { sheets?: number[] },
): Promise<LabeledBooleanResult>;
export async function booleanDifference(
  m0: Mesh, m1: Mesh, opts?: { returnCurves?: true; sheets?: number[] },
): Promise<LabeledBooleanResult | LabeledBooleanResultWithCurves> {
  assertSameDtype([m0, m1], ["mesh0", "mesh1"]);
  const dt = m0.dtype;
  const sheets = toSheets(opts?.sheets);
  if (opts?.returnCurves) {
    return dispatcher().run(
      () => native()[`dispatch_boolean_difference_with_curves_${dt}`](m0._handle, m1._handle, sheets),
      (raw) => wrapLabeledBooleanWithCurves(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_boolean_difference_${dt}`](m0._handle, m1._handle, sheets),
    (raw) => wrapLabeledBoolean(raw, dt),
  );
}

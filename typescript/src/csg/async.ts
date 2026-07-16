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
import {
  CsgGraph,
  CsgGraphOptions,
  CsgDomainsOptions,
  CsgMeshLabeledResult,
  CsgMeshIndexMapResult,
  CsgDomainsResult,
  CsgDomainsLabeledResult,
  CsgDomainsIndexMapResult,
  OuterShellOptions,
  buildCsgConfig,
  buildDomainsConfig,
  buildOuterShellConfig,
  splitExprArgs,
  wrapCsgMeshLabeled,
  wrapCsgMeshIndexMap,
  wrapCsgDomains,
  wrapCsgDomainsLabeled,
  wrapCsgDomainsIndexMap,
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

/** The boolean result mesh for `expr`, evaluated on a worker; with no
 * expression, the full arrangement mesh. `returnIndexMap` requires an
 * expression. */
export async function csgMesh(graph: CsgGraph): Promise<Mesh>;
export async function csgMesh(graph: CsgGraph, expr: Expr | number): Promise<Mesh>;
export async function csgMesh(
  graph: CsgGraph, opts: { returnSourceIds: true },
): Promise<CsgMeshLabeledResult>;
export async function csgMesh(
  graph: CsgGraph, expr: Expr | number, opts: { returnSourceIds: true },
): Promise<CsgMeshLabeledResult>;
export async function csgMesh(
  graph: CsgGraph, expr: Expr | number, opts: { returnIndexMap: true },
): Promise<CsgMeshIndexMapResult>;
export async function csgMesh(
  graph: CsgGraph,
  exprOrOpts?: Expr | number | { returnSourceIds?: true; returnIndexMap?: true },
  maybeOpts?: { returnSourceIds?: true; returnIndexMap?: true },
): Promise<Mesh | CsgMeshLabeledResult | CsgMeshIndexMapResult> {
  const [expr, opts] = splitExprArgs(exprOrOpts, maybeOpts);
  const program = expr === undefined ? [] : programOf(expr);
  const dt = graph.dtype;
  if (opts?.returnSourceIds && opts?.returnIndexMap) {
    throw new Error("returnSourceIds and returnIndexMap are exclusive");
  }
  if (opts?.returnIndexMap) {
    return dispatcher().run(
      () => native()[`dispatch_csg_mesh_with_index_map_${dt}`](graph._handle, program),
      (raw) => wrapCsgMeshIndexMap(raw, dt),
    );
  }
  if (opts?.returnSourceIds) {
    return dispatcher().run(
      () => native()[`dispatch_csg_mesh_with_labels_${dt}`](graph._handle, program),
      (raw) => wrapCsgMeshLabeled(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_csg_mesh_${dt}`](graph._handle, program),
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

/** Kept volumetric domains, extracted on a worker. The expression may be
 * omitted: `csgDomains(g)`, `csgDomains(g, opts)`, `csgDomains(g, expr)`,
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
  const cfg = buildDomainsConfig(opts);
  const dt = graph.dtype;
  if (opts?.returnSourceIds && opts?.returnIndexMap) {
    throw new Error("returnSourceIds and returnIndexMap are exclusive");
  }
  if (opts?.returnIndexMap) {
    return dispatcher().run(
      () => native()[`dispatch_csg_domains_with_index_map_${dt}`](graph._handle, program, cfg),
      (raw) => wrapCsgDomainsIndexMap(raw, dt),
    );
  }
  if (opts?.returnSourceIds) {
    return dispatcher().run(
      () => native()[`dispatch_csg_domains_with_labels_${dt}`](graph._handle, program, cfg),
      (raw) => wrapCsgDomainsLabeled(raw, dt),
    );
  }
  return dispatcher().run(
    () => native()[`dispatch_csg_domains_${dt}`](graph._handle, program, cfg),
    (raw) => wrapCsgDomains(raw, dt),
  );
}

/**
 * Repair a mesh to its outer shell on a worker: the boundary of the union
 * of everything it encloses. Same semantics as the sync `outerShell`.
 */
export async function outerShell(
  mesh: Mesh, opts?: OuterShellOptions,
): Promise<Mesh> {
  const dt = mesh.dtype;
  const cfg = buildOuterShellConfig(opts);
  return dispatcher().run(
    () => native()[`dispatch_outer_shell_${dt}`](mesh._handle, cfg),
    (raw) => new Mesh(raw, dt),
  );
}

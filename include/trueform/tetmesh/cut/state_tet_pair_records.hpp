/*
 * Copyright (c) 2026 XLAB
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
#pragma once
#include "../../arrangement/operands/complete_operands.hpp"
#include "../../core/buffer.hpp"
#include "../../exact/pt_converter.hpp"
#include "../../intersect/identity/materialize_intersection_points.hpp"
#include "../../intersect/intersect_config.hpp"
#include "../../intersect/intersect_mode.hpp"
#include "./tet_pair_workspace.hpp"
#include <cstddef>

namespace tf::tetmesh {

/// State the records of the surface triangles `ids` read as one form on
/// the lattice `points`, `triangles` naming their corners, into
/// `work.records` — the intersect module's own statement of every contact
/// among them (@ref tf::polygon_intersections), on `identity` in the house
/// mode with the form's own self records, at the band `band` — with the
/// form's points and faces
/// mapped to the surface's in `work.global` and `work.form_triangles`, and,
/// where `materialize` asks, each point's lattice position in
/// `work.positions`.
template <typename Index, typename Int, typename Points, typename Triangles,
          typename Ids>
auto state_tet_pair_records(
    const Points &points, const Triangles &triangles, const Ids &ids,
    const tf::exact::pt_converter<Int, Int, 3> &identity, double band,
    bool materialize, tet_pair_workspace<Index, Int> &work) -> void {
  auto &local = work.local;
  auto &global = work.global;
  if (local.size() < points.size()) {
    const std::size_t known = local.size();
    local.reallocate(points.size());
    for (std::size_t v = known; v < local.size(); ++v)
      local[v] = Index(-1);
  }
  global.clear();
  work.form_triangles.clear();
  auto &form = work.form;
  form.faces_buffer().allocate(ids.size());
  std::size_t f = 0;
  for (auto t : ids) {
    work.form_triangles.push_back(Index(t));
    for (std::size_t k = 0; k < 3; ++k) {
      const auto v = std::size_t(triangles[std::size_t(t)][k]);
      if (local[v] < 0) {
        local[v] = Index(global.size());
        global.push_back(Index(v));
      }
      form.faces_buffer()[f][k] = local[v];
    }
    ++f;
  }
  form.points_buffer().allocate(global.size());
  for (std::size_t v = 0; v < global.size(); ++v) {
    form.points_buffer()[v] = points[std::size_t(global[v])];
    local[std::size_t(global[v])] = Index(-1);
  }
  const auto completed = tf::arrangement::complete_operands(form.polygons());
  const tf::intersect_config intersect{
      tf::intersect_mode::primitives | tf::intersect_mode::within, band};
  const auto lattice =
      completed.template make_lattice<Int>(identity, intersect.tolerance);
  work.records = tf::polygon_intersections<Index, Int, Int>{};
  work.records.with_edge_splits(false);
  completed.build_intersections(work.records, lattice, intersect);
  if (!materialize)
    return;
  const auto reader = lattice.reader(completed.make_apply_to_form());
  tf::intersect::materialize_intersection_parameters<Index, Int>(work.records,
                                                                 work.dyadic);
  tf::intersect::materialize_intersection_points<Index, Int>(
      work.records,
      [&reader](Index flat) { return reader.point_of_flat(flat); }, work.dyadic,
      work.positions);
}

} // namespace tf::tetmesh

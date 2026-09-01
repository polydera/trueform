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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/dyadic_blend.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/wide_dyadic_ratio.hpp"
#include "../../exact/vertex.hpp"

#include <cstddef>

namespace tf::intersect {

/// Materialize every kind-E point's dyadic parameter on its home
/// carrier — the one ratio-to-dyadic conversion in the pipeline, once
/// per id. The dyadic IS the point's identity below the exact tier, so
/// this table is the primary materialization; positions are cheap
/// blends off it. Vertex-anchored ids occupy their slots with 0 so the
/// table is indexed by id directly.
template <typename Index, typename Int, typename Ibp>
auto materialize_intersection_parameters(
    const Ibp &ibp,
    tf::buffer<typename tf::exact::meta<Int>::param_type> &dyadic) -> void {
  const auto n = ibp.n_points();
  const auto n_vertex = ibp.n_vertex_points();
  dyadic.allocate(std::size_t(n));
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&](Index id) {
        if (id < n_vertex) {
          dyadic[std::size_t(id)] = 0;
          return;
        }
        const auto &t = ibp.exact_parameter(id);
        dyadic[std::size_t(id)] =
            tf::exact::wide_dyadic_ratio<Int>(t.num, t.den);
      },
      tf::checked(8192));
}

/// Materialize every canonical point's lattice position — the one
/// rounding in the pipeline, once per id.
///
/// A kind-V point IS an original vertex, so its position is copied; a
/// kind-E point blends its materialized dyadic parameter between the
/// carrier's endpoints. Every consumer reads this table, so a point has
/// one position however many carriers hold it.
///
/// `get_flat_point` resolves a flat vertex id to its converted lattice
/// position (frames applied, the kernels' own view of the originals).
template <typename Index, typename Int, typename Ibp, typename GetFlatPoint>
auto materialize_intersection_points(
    const Ibp &ibp, const GetFlatPoint &get_flat_point,
    const tf::buffer<typename tf::exact::meta<Int>::param_type> &dyadic,
    tf::buffer<tf::exact::pt3<Int>> &points) -> void {
  const auto vertex_offsets = ibp.vertex_offsets();
  const auto n = ibp.n_points();
  const auto n_vertex = ibp.n_vertex_points();
  points.allocate(std::size_t(n));
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&](Index id) {
        if (id < n_vertex) {
          const auto &anchor = ibp.vertex_anchor(id);
          points[std::size_t(id)] = get_flat_point(
              vertex_offsets[std::size_t(anchor.tag)] + anchor.vid);
          return;
        }
        const auto &edge = ibp.home_edge(id);
        const auto u = get_flat_point(edge.u);
        const auto v = get_flat_point(edge.v);
        const auto dy = dyadic[std::size_t(id)];
        tf::exact::pt3<Int> p;
        for (int d = 0; d < 3; ++d)
          p[d] = tf::exact::dyadic_blend(u[d], v[d], dy);
        points[std::size_t(id)] = p;
      },
      tf::checked(8192));
}

} // namespace tf::intersect

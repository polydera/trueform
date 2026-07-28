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

#include "../../exact/meta.hpp"
#include "../../exact/rebase_parameter.hpp"
#include "../../exact/snap_parameter_to_edge.hpp"
#include "../detail/region_triangulation_identity.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "./constraint_parent_span.hpp"
#include <cstddef>

namespace tf::cut {

/// @ingroup cut
/// @brief Identify every point a resolving build created with the existing
///        vertex its crossing parameter names.
///
/// A crossing the wave can transport becomes a split. What still refuses
/// after the wave is a crossing whose position on the transported grid IS an
/// existing vertex — one snapped step is one ulp of the source real along
/// the edge, so there is no split to make and asking again preserved would
/// refuse forever. The resolving build had to create a point to complete,
/// but the point is not a new fact: it is the vertex its parameter rounds
/// onto, judged through the same snap and the same one-quantum tolerance
/// that define split identity everywhere else. Writing that identification
/// into `rep` is what lets the completed triangulation be kept.
///
/// One parent naming an endpoint identifies the point. The other parent's
/// parameter is measured along a nearly parallel piece at grazing
/// incidence, where sub-lattice rounding of the piece's own endpoints
/// amplifies into an arbitrary reading along the edge — a crossing
/// genuinely interior to that parent is a T-junction split, which the wave
/// files long before this runs. When both parents name endpoints and they
/// differ, those two vertices are one point under this precision, and the
/// weld is reported like any other merge. A created point no parent can
/// name is left unmapped, so the emit refuses.
template <typename Index, typename Int, typename VertexAt, typename GetPoint>
auto identify_resolved_crossings(
    tf::cut::detail::region_triangulation_workspace<Index, Int> &workspace,
    Index n_tags, Index tag, Index n_input, const VertexAt &vertex_at,
    const GetPoint &get_point) -> void {
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto &input_to_output = workspace.tri.index_map().f();
  const auto &output_to_input = workspace.tri.index_map().kept_ids();
  auto endpoint_named_by = [&](Index constraint,
                               param_t parameter) -> Index {
    const auto span = tf::cut::constraint_parent_span<Index, Int>(
        workspace, n_tags, tag, std::size_t(constraint));
    const param_t physical = tf::exact::rebase_parameter<Int>(
        parameter, span.t0, span.t1,
        typename tf::exact::meta<Int>::T2(1)
            << tf::exact::meta<Int>::split_grid_bits);
    if (physical < param_t(0))
      return Index(-1);
    const bool forward =
        tf::cut::detail::region_vertex_key(n_tags, tag, span.edge[0]) <
        tf::cut::detail::region_vertex_key(n_tags, tag, span.edge[1]);
    const auto &low = forward ? span.edge[0] : span.edge[1];
    const auto &high = forward ? span.edge[1] : span.edge[0];
    const param_t snapped = tf::exact::snap_parameter_to_edge<Int>(
        physical, get_point(low), get_point(high));
    // Same-split tolerance as tf::cut::has_adjacent_region_split: one
    // quantum of the grid the transported parameter defines.
    for (int end = 0; end < 2; ++end) {
      const param_t distance = snapped - (end == 0 ? span.t0 : span.t1);
      if (distance <= param_t(1) && distance >= param_t(-1))
        return workspace.cons[std::size_t(2 * constraint + end)];
    }
    return Index(-1);
  };
  for (const auto &record : workspace.tri.parameterized_crossings()) {
    if (record.id_b == Index(-1) ||
        output_to_input[std::size_t(record.point)] != n_input)
      continue;
    const Index on_first = endpoint_named_by(record.id_a, record.t_a);
    const Index on_second = endpoint_named_by(record.id_b, record.t_b);
    if (on_first == Index(-1) && on_second == Index(-1))
      continue;
    Index kept = on_first != Index(-1) ? on_first : on_second;
    if (on_first != Index(-1) && on_second != Index(-1) &&
        tf::cut::detail::region_vertex_key(
            n_tags, tag, vertex_at(std::size_t(on_second))) <
            tf::cut::detail::region_vertex_key(
                n_tags, tag, vertex_at(std::size_t(on_first))))
      kept = on_second;
    const Index canonical =
        workspace.rep[std::size_t(input_to_output[std::size_t(kept)])];
    workspace.rep[std::size_t(record.point)] = canonical;
    const Index other =
        on_first == on_second ? Index(-1)
                              : (kept == on_first ? on_second : on_first);
    if (other != Index(-1)) {
      const auto other_key = tf::cut::detail::region_vertex_key(
          n_tags, tag, vertex_at(std::size_t(other)));
      const auto kept_key = tf::cut::detail::region_vertex_key(
          n_tags, tag, vertex_at(std::size_t(canonical)));
      if (other_key != kept_key)
        workspace.merges.push_back(
            {other_key, kept_key, vertex_at(std::size_t(canonical))});
    }
  }
}

} // namespace tf::cut

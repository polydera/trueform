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
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_identity.hpp"
#include "../detail/region_triangulation_types.hpp"
#include <array>
#include <cstddef>

namespace tf::cut {

/// @ingroup cut
/// @brief The edge a constraint belongs to, and the span it occupies on it.
///
/// A split has to be registered against the edge every carrier of it shares,
/// never against the piece one carrier happened to be handed. The two
/// coincide only until that edge is first split; afterwards a record filed
/// against a piece is keyed by endpoints no walk ever forms, so it is
/// invisible to every carrier including the region that reported it, and the
/// next round rediscovers the same split and dedups it against a record
/// nothing can see.
template <typename Index, typename Int> struct constraint_parent {
  std::array<tf::intersect::graph::vertex<Index>, 2> edge;
  typename tf::exact::meta<Int>::param_type t0;
  typename tf::exact::meta<Int>::param_type t1;
};

/// @ingroup cut
/// @brief Resolve constraint `edge` of `workspace` to @ref
///        tf::cut::constraint_parent.
///
/// `t0`/`t1` bound the piece on its parent, oriented the way the parent's
/// own key order runs, so a position measured on the piece rebases onto the
/// parent by interpolating between them. A whole constraint spans its parent
/// entirely and reports the full range in that same orientation.
template <typename Index, typename Int>
auto constraint_parent_span(
    const tf::cut::detail::region_triangulation_workspace<Index, Int>
        &workspace,
    Index n_tags, Index tag, std::size_t edge)
    -> tf::cut::constraint_parent<Index, Int> {
  using param_t = typename tf::exact::meta<Int>::param_type;
  const bool sub = bool(workspace.cons_sub[edge]);
  const auto &parent =
      sub ? workspace.cons_parent[edge] : workspace.cons_verts[edge];
  if (sub)
    return {parent, workspace.cons_t[edge][0], workspace.cons_t[edge][1]};
  const param_t maximum = param_t(1) << tf::exact::meta<Int>::split_grid_bits;
  const bool forward =
      tf::cut::detail::region_vertex_key(n_tags, tag, parent[0]) <
      tf::cut::detail::region_vertex_key(n_tags, tag, parent[1]);
  if (forward)
    return {parent, param_t(0), maximum};
  return {parent, maximum, param_t(0)};
}

} // namespace tf::cut

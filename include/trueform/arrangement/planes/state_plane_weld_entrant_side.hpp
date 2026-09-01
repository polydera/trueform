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

#include "../../core/buffer.hpp"
#include "../../intersect/graph/plane_def_respan.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include "./plane_world.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement {

/// Emit one raw source side in the immutable LA's final edge currency.
///
/// A promoted face starts from its original corner cycle, while PA consumes
/// LA's already closed table. A side on an LA split root therefore contributes
/// every surviving piece of that root; any LA weld is applied to the two
/// endpoints. The source side remains the authority for face, ordinal,
/// boundary and radial metadata.
template <typename Index, typename Policy, typename VertexOffsets>
auto state_plane_weld_entrant_side(
    const plane_world<Policy> &world, const VertexOffsets &vertex_offsets,
    const tf::intersect::graph::plane_edge_def<Index> &parent,
    tf::buffer<tf::intersect::graph::plane_edge_def<Index>> &out) -> void {
  const auto &graph = world.graph();
  const auto la_merges = world.merges();
  const auto ends = tf::intersect::graph::plane_merged_endpoints<Index>(
      la_merges, vertex_offsets, parent);
  std::array<Index, 2> current{Index(ends.lo_tag), ends.lo_id};
  const std::array<Index, 2> last{Index(ends.hi_tag), ends.hi_id};

  const auto canonical =
      tf::intersect::graph::find_plane_canon_group(graph, parent);
  const auto roots = world.split_roots();
  const auto split =
      canonical == Index(-1)
          ? roots.end()
          : std::lower_bound(roots.begin(), roots.end(), canonical);
  if (split == roots.end() || *split != canonical) {
    if (current == last)
      return;
    auto piece = parent;
    tf::intersect::graph::plane_piece_def(piece, current, last, parent,
                                           Index(1));
    out.push_back(piece);
    return;
  }

  const auto group = std::size_t(split - roots.begin());
  const auto survivors = world.split_survivors(group);
  const auto count = Index(survivors.size()) + Index(1);
  for (Index position = 0; position < count; ++position) {
    const auto stop =
        position + Index(1) == count
            ? last
            : std::array<Index, 2>{Index(-1), survivors[std::size_t(position)]};
    if (current != stop) {
      auto piece = parent;
      tf::intersect::graph::plane_piece_def(piece, current, stop, parent,
                                             count);
      out.push_back(piece);
    }
    current = stop;
  }
}

} // namespace tf::arrangement

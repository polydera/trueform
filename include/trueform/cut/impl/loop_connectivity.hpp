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

#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
#include "../detail/compute_region_connectivity.hpp"
#include "../detail/filter_dead_connectivity.hpp"
#include "../face_cuts.hpp"
#include "../face_regions.hpp"

namespace tf::cut {

/// @brief Per-carrier per-directed-edge neighboring carriers.
///
/// Regions contribute boundary edges followed by hole-walk edges. The supplied
/// dead-carrier mask filters rows and references. Identity is topological only.
template <typename Index> class loop_connectivity {
public:
  loop_connectivity() = default;

  auto n_tags() const -> Index {
    return _tag_carrier_offsets.size()
               ? static_cast<Index>(_tag_carrier_offsets.size() - 1)
               : Index(0);
  }

  auto connectivity_per_carrier_edge() const {
    return tf::make_offset_block_range(
        _carrier_edge_offsets,
        tf::make_offset_block_range(_edge_neighbour_offsets,
                                    tf::make_range(_edge_neighbours)));
  }

  auto clear() -> void {
    _tag_carrier_offsets.clear();
    _carrier_edge_offsets.clear();
    _edge_neighbour_offsets.clear();
    _edge_neighbours.clear();
  }

  template <typename Int, typename DeadRange, typename Counts>
  auto build(const tf::face_regions<Index, Int> &face_regions,
             const DeadRange &dead_regions, const Counts &point_counts,
             Index n_created) -> void {
    clear();
    tf::cut::detail::compute_region_connectivity(
        face_regions, point_counts, n_created, _carrier_edge_offsets,
        _edge_neighbour_offsets, _edge_neighbours);
    finish_build(dead_regions, face_regions.tag_offsets());
  }

private:
  template <typename DeadRange, typename TagOffsets>
  auto finish_build(const DeadRange &dead_carriers,
                    const TagOffsets &tag_offsets) -> void {
    bool any_dead = false;
    for (auto dead : dead_carriers)
      if (dead) {
        any_dead = true;
        break;
      }
    if (any_dead)
      tf::cut::detail::filter_dead_connectivity(
          dead_carriers, _carrier_edge_offsets, _edge_neighbour_offsets,
          _edge_neighbours);

    _tag_carrier_offsets.allocate(tag_offsets.size());
    tf::parallel_copy(tag_offsets, tf::make_range(_tag_carrier_offsets));
  }

  tf::buffer<Index> _tag_carrier_offsets;
  tf::buffer<Index> _carrier_edge_offsets;
  tf::buffer<Index> _edge_neighbour_offsets;
  tf::buffer<Index> _edge_neighbours;
};

} // namespace tf::cut

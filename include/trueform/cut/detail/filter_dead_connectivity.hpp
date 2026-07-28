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
#pragma once

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"

#include <cstddef>
#include <numeric>
#include <utility>

namespace tf::cut::detail {

template <typename Index, typename Mask>
auto filter_dead_connectivity(const Mask &dead_carriers,
                              tf::buffer<Index> &carrier_edge_offsets,
                              tf::buffer<Index> &edge_neighbour_offsets,
                              tf::buffer<Index> &edge_neighbours) -> void {
  const Index n_carriers = static_cast<Index>(carrier_edge_offsets.size() - 1);
  if (n_carriers == 0)
    return;

  tf::buffer<Index> new_carrier_edge_offsets;
  new_carrier_edge_offsets.allocate(static_cast<std::size_t>(n_carriers + 1));
  new_carrier_edge_offsets[0] = Index(0);
  tf::parallel_for_each(
      tf::make_sequence_range(n_carriers),
      [&](Index carrier_id) {
        new_carrier_edge_offsets[carrier_id + 1] =
            dead_carriers[carrier_id]
                ? Index(0)
                : Index(carrier_edge_offsets[carrier_id + 1] -
                        carrier_edge_offsets[carrier_id]);
      },
      tf::checked);
  std::partial_sum(new_carrier_edge_offsets.begin(),
                   new_carrier_edge_offsets.end(),
                   new_carrier_edge_offsets.begin());
  const Index new_n_edges = new_carrier_edge_offsets[n_carriers];

  tf::buffer<Index> new_edge_neighbour_offsets;
  new_edge_neighbour_offsets.allocate(
      static_cast<std::size_t>(new_n_edges + 1));
  new_edge_neighbour_offsets[0] = Index(0);
  tf::parallel_for_each(
      tf::make_sequence_range(n_carriers), [&](Index carrier_id) {
        if (dead_carriers[carrier_id])
          return;
        const Index old_first = carrier_edge_offsets[carrier_id];
        const Index old_last = carrier_edge_offsets[carrier_id + 1];
        Index new_edge_id = new_carrier_edge_offsets[carrier_id];
        for (Index edge_id = old_first; edge_id < old_last;
             ++edge_id, ++new_edge_id) {
          const Index neighbour_first = edge_neighbour_offsets[edge_id];
          const Index neighbour_last = edge_neighbour_offsets[edge_id + 1];
          Index live_count = 0;
          for (Index i = neighbour_first; i < neighbour_last; ++i)
            if (!dead_carriers[edge_neighbours[i]])
              ++live_count;
          new_edge_neighbour_offsets[new_edge_id + 1] = live_count;
        }
      });
  std::partial_sum(new_edge_neighbour_offsets.begin(),
                   new_edge_neighbour_offsets.end(),
                   new_edge_neighbour_offsets.begin());
  const Index new_n_neighbours = new_edge_neighbour_offsets[new_n_edges];

  tf::buffer<Index> new_edge_neighbours;
  new_edge_neighbours.allocate(static_cast<std::size_t>(new_n_neighbours));
  tf::parallel_for_each(
      tf::make_sequence_range(n_carriers), [&](Index carrier_id) {
        if (dead_carriers[carrier_id])
          return;
        const Index old_first = carrier_edge_offsets[carrier_id];
        const Index old_last = carrier_edge_offsets[carrier_id + 1];
        Index new_edge_id = new_carrier_edge_offsets[carrier_id];
        for (Index edge_id = old_first; edge_id < old_last;
             ++edge_id, ++new_edge_id) {
          const Index neighbour_first = edge_neighbour_offsets[edge_id];
          const Index neighbour_last = edge_neighbour_offsets[edge_id + 1];
          Index write_id = new_edge_neighbour_offsets[new_edge_id];
          for (Index i = neighbour_first; i < neighbour_last; ++i) {
            const Index neighbour = edge_neighbours[i];
            if (!dead_carriers[neighbour])
              new_edge_neighbours[write_id++] = neighbour;
          }
        }
      });

  carrier_edge_offsets = std::move(new_carrier_edge_offsets);
  edge_neighbour_offsets = std::move(new_edge_neighbour_offsets);
  edge_neighbours = std::move(new_edge_neighbours);
}

} // namespace tf::cut::detail

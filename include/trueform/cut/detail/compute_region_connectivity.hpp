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

#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/faces.hpp"
#include "../../core/range.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../face_regions.hpp"
#include "./make_connectivity_face_membership.hpp"

#include <algorithm>
#include <cstddef>
#include <tuple>

namespace tf::cut::detail {
namespace region_connectivity {

template <typename Index, typename Loops, typename Holes, typename LoopHoles,
          typename Function>
auto for_each_walk(Index region_id, const Loops &loops, const Holes &holes,
                   const LoopHoles &loop_holes, const Function &function)
    -> bool {
  if (function(loops[region_id]))
    return true;
  for (auto hole_id : loop_holes[region_id])
    if (function(holes[hole_id]))
      return true;
  return false;
}

template <typename Index, typename Loops, typename Holes, typename LoopHoles>
auto count_edge_occurrences(Index candidate, Index v0, Index v1,
                            bool first_only, const Loops &loops,
                            const Holes &holes, const LoopHoles &loop_holes,
                            const tf::buffer<Index> &carrier_edge_offsets,
                            const tf::buffer<Index> &flat_ids) -> Index {
  Index count = 0;
  Index offset = carrier_edge_offsets[static_cast<std::size_t>(candidate)];
  for_each_walk(candidate, loops, holes, loop_holes, [&](const auto &walk) {
    const Index size = static_cast<Index>(walk.size());
    if (size) {
      Index previous = size - 1;
      for (Index i = 0; i < size; previous = i++) {
        const Index a = flat_ids[static_cast<std::size_t>(offset + previous)];
        const Index b = flat_ids[static_cast<std::size_t>(offset + i)];
        count += Index((char(a == v0) & char(b == v1)) |
                       (char(a == v1) & char(b == v0)));
        if (first_only && count)
          return true;
      }
    }
    offset += size;
    return false;
  });
  return count;
}

template <typename Index> struct local_output {
  tf::buffer<Index> offsets;
  tf::buffer<Index> neighbours;
};

template <typename Index>
auto aggregate(const local_output<Index> &local,
               std::tuple<tf::buffer<Index> &, tf::buffer<Index> &> result)
    -> void {
  auto &[offsets, neighbours] = result;
  const auto old_neighbour_count = neighbours.size();
  neighbours.reallocate(old_neighbour_count + local.neighbours.size());
  std::copy(local.neighbours.begin(), local.neighbours.end(),
            neighbours.begin() + old_neighbour_count);

  const auto old_offset_count = offsets.size();
  offsets.reallocate(old_offset_count + local.offsets.size());
  auto output = offsets.begin() + old_offset_count;
  for (auto offset : local.offsets)
    *output++ = offset + Index(old_neighbour_count);
}

} // namespace region_connectivity

/// Membership is per region, but edge wrapping is per walk; concatenating walks
/// would invent seam edges.
template <typename Index, typename Int, typename Counts>
auto compute_region_connectivity(
    const tf::face_regions<Index, Int> &face_regions,
    const Counts &point_counts, Index n_created,
    tf::buffer<Index> &carrier_edge_offsets,
    tf::buffer<Index> &edge_neighbour_offsets,
    tf::buffer<Index> &edge_neighbours) -> void {
  using source = tf::intersect::graph::vertex_source;

  const auto loops = face_regions.loops();
  const auto holes = face_regions.holes();
  const auto loop_holes = face_regions.loop_holes();
  const auto descriptors = face_regions.descriptors();
  const std::size_t n_regions = loops.size();
  if (!n_regions)
    return;

  const std::size_t n_tags = face_regions.tag_offsets().size() - 1;
  tf::buffer<Index> tag_base;
  tag_base.allocate(n_tags + 1);
  tag_base[0] = n_created;
  for (std::size_t tag = 0; tag < n_tags; ++tag)
    tag_base[tag + 1] = tag_base[tag] + static_cast<Index>(point_counts[tag]);

  carrier_edge_offsets.allocate(n_regions + 1);
  carrier_edge_offsets[0] = 0;
  for (std::size_t region_id = 0; region_id < n_regions; ++region_id) {
    Index count = 0;
    region_connectivity::for_each_walk(
        static_cast<Index>(region_id), loops, holes, loop_holes,
        [&](const auto &walk) {
          count += static_cast<Index>(walk.size());
          return false;
        });
    carrier_edge_offsets[region_id + 1] =
        carrier_edge_offsets[region_id] + count;
  }

  tf::buffer<Index> flat_ids;
  flat_ids.allocate(static_cast<std::size_t>(carrier_edge_offsets[n_regions]));
  tf::parallel_for_each(
      tf::make_sequence_range(n_regions), [&](std::size_t region_id) {
        const Index tag = descriptors[region_id].tag;
        const Index base = tag_base[static_cast<std::size_t>(tag)];
        Index offset = carrier_edge_offsets[region_id];
        region_connectivity::for_each_walk(
            static_cast<Index>(region_id), loops, holes, loop_holes,
            [&](const auto &walk) {
              for (const auto &vertex : walk)
                flat_ids[static_cast<std::size_t>(offset++)] =
                    vertex.source == source::created ? vertex.id
                                                     : base + vertex.id;
              return false;
            });
      });

  const Index bounded_id_count = tag_base[n_tags];
  auto region_faces = tf::make_offset_block_range(
      tf::make_range(carrier_edge_offsets), tf::make_range(flat_ids));
  auto membership = tf::cut::detail::make_connectivity_face_membership(
      tf::make_faces(region_faces), flat_ids, bounded_id_count);

  auto task = [&](auto &&range,
                  region_connectivity::local_output<Index> &local) {
    for (const auto &[region_id_z, region] : range) {
      const Index region_id = static_cast<Index>(region_id_z);
      Index offset = carrier_edge_offsets[static_cast<std::size_t>(region_id)];
      auto process_edge = [&](Index v0, Index v1) {
        local.offsets.push_back(static_cast<Index>(local.neighbours.size()));
        const auto &membership0 = membership[v0];
        const auto &membership1 = membership[v1];
        auto current0 = membership0.begin();
        const auto end0 = membership0.end();
        auto current1 = membership1.begin();
        const auto end1 = membership1.end();

        Index previous = region_id;
        Index emitted = 0;
        Index count = 0;
        bool fully_counted = false;
        while ((current0 != end0) & (current1 != end1)) {
          if (*current0 > *current1) {
            ++current0;
          } else {
            if (char(Index(*current0) != region_id) &
                char(!(*current1 > *current0))) {
              const Index candidate = Index(*current0);
              if (candidate != previous) {
                previous = candidate;
                emitted = 0;
                fully_counted = false;
                count = region_connectivity::count_edge_occurrences(
                    candidate, v0, v1, true, loops, holes, loop_holes,
                    carrier_edge_offsets, flat_ids);
              } else if (!fully_counted) {
                fully_counted = true;
                count = region_connectivity::count_edge_occurrences(
                    candidate, v0, v1, false, loops, holes, loop_holes,
                    carrier_edge_offsets, flat_ids);
              }
              if (emitted < count) {
                ++emitted;
                local.neighbours.push_back(candidate);
              }
              ++current0;
            }
            ++current1;
          }
        }
      };

      region_connectivity::for_each_walk(
          region_id, loops, holes, loop_holes, [&](const auto &walk) {
            const Index size = static_cast<Index>(walk.size());
            for (Index i = 0; i < size; ++i)
              process_edge(
                  flat_ids[static_cast<std::size_t>(offset + i)],
                  flat_ids[static_cast<std::size_t>(offset + (i + 1) % size)]);
            offset += size;
            return false;
          });
      (void)region;
    }
  };

  edge_neighbour_offsets.clear();
  edge_neighbours.clear();
  edge_neighbour_offsets.reserve(
      static_cast<std::size_t>(carrier_edge_offsets[n_regions]) + 1);
  tf::blocked_reduce_sequenced_aggregate(
      tf::enumerate(loops), std::tie(edge_neighbour_offsets, edge_neighbours),
      region_connectivity::local_output<Index>{}, task,
      region_connectivity::aggregate<Index>);
  edge_neighbour_offsets.push_back(static_cast<Index>(edge_neighbours.size()));
}

} // namespace tf::cut::detail

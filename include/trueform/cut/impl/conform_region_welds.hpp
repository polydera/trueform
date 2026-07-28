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
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../face_regions.hpp"
#include "./resolve_region_vertex.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace tf::cut {

template <typename Index, typename Int>
auto conform_region_welds(
    const tf::face_regions<Index, Int> &regions, Index n_tags,
    const tf::buffer<tf::cut::detail::region_triangulation_merge<Index>>
        &merges,
    const tf::buffer<tf::intersect::graph::face_descriptor<Index>>
        &promoted,
    tf::buffer<std::array<Index, 2>> &ranges,
    tf::buffer<
        std::array<tf::intersect::graph::vertex<Index>, 3>> &triangles)
    -> void {
  if (merges.size() == 0)
    return;
  auto descriptors = regions.descriptors();
  const std::size_t n_regions = descriptors.size();
  const std::size_t n_loops = ranges.size();
  tf::buffer<Index> counts;
  counts.allocate(n_loops);
  tf::parallel_for_each(
      tf::make_sequence_range(n_loops), [&](std::size_t loop) {
        const Index tag =
            loop < n_regions ? descriptors[loop].tag
                             : promoted[loop - n_regions].tag;
        const auto range = ranges[loop];
        Index count = 0;
        for (Index triangle = range[0]; triangle < range[1]; ++triangle) {
          auto &vertices = triangles[std::size_t(triangle)];
          for (int corner = 0; corner < 3; ++corner) {
            const auto &vertex = vertices[std::size_t(corner)];
            auto resolved =
                tf::cut::resolve_region_vertex(n_tags, tag, vertex, merges);
            if (resolved.source != vertex.source ||
                resolved.id != vertex.id)
              resolved.sub_id = {0, tf::topo_type::face};
            vertices[std::size_t(corner)] = resolved;
          }
          const auto key_0 = tf::cut::resolve_region_vertex_key(
              n_tags, tag, vertices[0], merges);
          const auto key_1 = tf::cut::resolve_region_vertex_key(
              n_tags, tag, vertices[1], merges);
          const auto key_2 = tf::cut::resolve_region_vertex_key(
              n_tags, tag, vertices[2], merges);
          if (!(key_0 == key_1 || key_1 == key_2 ||
                key_0 == key_2))
            ++count;
        }
        counts[loop] = count;
      });

  tf::buffer<std::array<Index, 2>> compact_ranges;
  compact_ranges.allocate(n_loops);
  Index offset = 0;
  for (std::size_t loop = 0; loop < n_loops; ++loop) {
    compact_ranges[loop] = {offset, offset + counts[loop]};
    offset += counts[loop];
  }
  tf::buffer<
      std::array<tf::intersect::graph::vertex<Index>, 3>>
      compact_triangles;
  compact_triangles.allocate(std::size_t(offset));
  tf::parallel_for_each(
      tf::make_sequence_range(n_loops), [&](std::size_t loop) {
        const Index tag =
            loop < n_regions ? descriptors[loop].tag
                             : promoted[loop - n_regions].tag;
        const auto range = ranges[loop];
        Index write = compact_ranges[loop][0];
        for (Index triangle = range[0]; triangle < range[1]; ++triangle) {
          const auto &vertices = triangles[std::size_t(triangle)];
          const auto key_0 = tf::cut::resolve_region_vertex_key(
              n_tags, tag, vertices[0], merges);
          const auto key_1 = tf::cut::resolve_region_vertex_key(
              n_tags, tag, vertices[1], merges);
          const auto key_2 = tf::cut::resolve_region_vertex_key(
              n_tags, tag, vertices[2], merges);
          if (key_0 == key_1 || key_1 == key_2 || key_0 == key_2)
            continue;
          compact_triangles[std::size_t(write++)] = vertices;
        }
      });
  triangles = std::move(compact_triangles);
  ranges = std::move(compact_ranges);
}

} // namespace tf::cut

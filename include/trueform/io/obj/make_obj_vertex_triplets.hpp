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

#include "bucket_obj_corners_by_position.hpp"
#include "obj_corner_attributes.hpp"

#include "../../core/algorithm/mask_to_index_map.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/index_map.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::io::obj {

/// @brief Names one output vertex per distinct face-corner triplet.
///
/// A corner names a position, and a position is a dense bounded id, so the
/// corners sharing one are counts plus one prefix. Inside a position only the
/// attribute pair can still separate corners, and those records sort by value;
/// a file that names no attributes has none to separate them by, so it stops
/// at the referenced positions.
///
/// The vertices come out ordered by `(position, texture, normal)`.
///
/// @param corner_positions The position every corner names.
/// @param corner_attributes The `(texture, normal)` every corner names, empty
/// when the file's faces name neither.
/// @param n_positions The size of the position id space.
/// @param chunk_count How many corner chunks carry the grouping passes.
/// @param corner_vertices Filled with the vertex each corner resolves to.
/// @param vertices Filled with the triplet each vertex stands for.
template <typename Index>
auto make_obj_vertex_triplets(
    const tf::buffer<int> &corner_positions,
    const tf::buffer<std::array<int, 2>> &corner_attributes,
    std::size_t n_positions, std::size_t chunk_count,
    tf::buffer<Index> &corner_vertices,
    tf::buffer<std::array<int, 3>> &vertices) -> void {
  const auto n_corners = corner_positions.size();
  corner_vertices.allocate(n_corners);

  if (corner_attributes.size() == 0) {
    tf::buffer<bool> referenced;
    referenced.allocate(n_positions);
    tf::parallel_fill(referenced, false);
    // benign race: threads write `true` to the same byte, and the parallel
    // loop ends before the mask is read.
    tf::parallel_for_each(
        corner_positions, [&](int position) { referenced[position] = true; },
        tf::checked);
    auto positions = tf::mask_to_index_map<Index>(referenced);
    tf::parallel_for_each(
        tf::enumerate(corner_positions),
        [&](auto pair) {
          auto &&[corner, position] = pair;
          corner_vertices[corner] = positions.f()[position];
        },
        tf::checked);
    vertices.allocate(positions.kept_ids().size());
    tf::parallel_for_each(
        tf::enumerate(positions.kept_ids()),
        [&](auto pair) {
          auto &&[vertex, position] = pair;
          vertices[vertex] = {static_cast<int>(position), -1, -1};
        },
        tf::checked);
    return;
  }

  tf::buffer<Index> offsets;
  tf::buffer<obj_corner_attributes<Index>> records;
  bucket_obj_corners_by_position(corner_positions, corner_attributes,
                                 n_positions, chunk_count, offsets, records);

  tf::buffer<Index> bases;
  bases.allocate(n_positions);
  tf::parallel_for_each(
      tf::make_sequence_range(n_positions),
      [&](std::size_t position) {
        auto *begin = records.begin() + offsets[position];
        auto *end = records.begin() + offsets[position + 1];
        std::sort(begin, end);
        Index distinct = 0;
        for (auto *slot = begin; slot != end; ++slot)
          if (slot == begin || !names_one_obj_vertex(*(slot - 1), *slot))
            ++distinct;
        bases[position] = distinct;
      },
      tf::checked);

  Index n_vertices = 0;
  for (std::size_t position = 0; position < n_positions; ++position) {
    const auto count = bases[position];
    bases[position] = n_vertices;
    n_vertices += count;
  }
  vertices.allocate(static_cast<std::size_t>(n_vertices));
  tf::parallel_for_each(
      tf::make_sequence_range(n_positions),
      [&](std::size_t position) {
        auto *begin = records.begin() + offsets[position];
        auto *end = records.begin() + offsets[position + 1];
        auto vertex = bases[position];
        for (auto *slot = begin; slot != end; ++slot) {
          if (slot != begin && !names_one_obj_vertex(*(slot - 1), *slot))
            ++vertex;
          vertices[vertex] = {static_cast<int>(position), slot->texture,
                              slot->normal};
          corner_vertices[slot->corner] = vertex;
        }
      },
      tf::checked);
}

} // namespace tf::io::obj

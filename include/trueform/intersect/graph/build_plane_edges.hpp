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
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::intersect::graph {

/// plane -> its members' definitions: the counts are the members' own
/// emission spans, so one prefix hands every plane a disjoint slice.
/// Ordering that slice by definition index IS canonical order, so a
/// plane's block arrives grouped by canonical edge — the only order
/// its consumers read it in — and the ordering runs per block because
/// that is where the work is: a handful of definitions, not the whole
/// table.
template <typename Index, typename PlaneFaces>
auto build_plane_edges(Index n_planes, const PlaneFaces &plane_faces,
                       const tf::buffer<Index> &face_def_offsets,
                       const tf::buffer<Index> &emission_to_canon,
                       tf::offset_block_buffer<Index, Index> &edges) -> void {
  auto &offsets = edges.offsets_buffer();
  auto &data = edges.data_buffer();
  offsets.allocate(std::size_t(n_planes) + 1);
  offsets[0] = 0;
  tf::parallel_for_each(
      tf::make_sequence_range(n_planes),
      [&](Index p) {
        Index count = 0;
        for (const auto f : plane_faces[std::size_t(p)])
          count += face_def_offsets[std::size_t(f) + 1] -
                   face_def_offsets[std::size_t(f)];
        offsets[std::size_t(p) + 1] = count;
      },
      tf::checked);
  for (std::size_t i = 1; i <= std::size_t(n_planes); ++i)
    offsets[i] += offsets[i - 1];
  data.allocate(std::size_t(offsets[std::size_t(n_planes)]));
  tf::parallel_for_each(
      tf::make_sequence_range(n_planes),
      [&](Index p) {
        const auto begin = std::size_t(offsets[std::size_t(p)]);
        auto w = begin;
        for (const auto f : plane_faces[std::size_t(p)])
          for (auto e = std::size_t(face_def_offsets[std::size_t(f)]);
               e < std::size_t(face_def_offsets[std::size_t(f) + 1]); ++e)
            data[w++] = emission_to_canon[e];
        std::sort(data.begin() + begin, data.begin() + w);
      },
      tf::checked);
}

} // namespace tf::intersect::graph

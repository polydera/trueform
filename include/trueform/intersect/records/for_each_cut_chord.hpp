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
#include "../../core/coordinate_dims.hpp"
#include "../../core/range.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/slice.hpp"
#include "../../topology/topo_type.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::intersect {

/// Emit the chords of one (face, cut) record group.
///
/// Calls `emit(id0, id1, on_boundary)` for every chord. Vertex hits on
/// adjacent slots bound a mesh edge lying on the cut value and pair as
/// boundary chords; the remaining points cross the boundary transversally,
/// so sorted along the dominant spread they alternate inside/outside and
/// consecutive pairs are the interior chords.
///
/// Records share one face and cut and are sorted by (target.label,
/// target.id), so vertex hits form a slot-ordered prefix.
template <typename Index, typename Iterator0, std::size_t N0,
          typename Iterator1, std::size_t N1, typename Emit>
auto for_each_cut_chord(const tf::range<Iterator0, N0> &records,
                        Index face_size,
                        const tf::range<Iterator1, N1> &points,
                        const Emit &emit) -> void {
  std::size_t vertex_end = 0;
  while (vertex_end < records.size() &&
         records[vertex_end].target.label == tf::topo_type::vertex)
    ++vertex_end;

  tf::small_vector<bool, 8> in_flat;
  in_flat.resize(vertex_end);
  for (std::size_t i = 0; i + 1 < vertex_end; ++i)
    if (Index(records[i].target.id) + 1 == Index(records[i + 1].target.id)) {
      emit(records[i].id, records[i + 1].id, true);
      in_flat[i] = true;
      in_flat[i + 1] = true;
    }
  if (vertex_end >= 2 && Index(records[0].target.id) == 0 &&
      Index(records[vertex_end - 1].target.id) + 1 == face_size) {
    emit(records[vertex_end - 1].id, records[0].id, true);
    in_flat[0] = true;
    in_flat[vertex_end - 1] = true;
  }

  tf::small_vector<Index, 8> transversal;
  for (std::size_t i = 0; i < vertex_end; ++i)
    if (!in_flat[i])
      transversal.push_back(records[i].id);
  for (std::size_t i = vertex_end; i < records.size(); ++i)
    transversal.push_back(records[i].id);
  if (transversal.size() < 2)
    return;

  if (transversal.size() > 2) {
    constexpr std::size_t Dims = tf::coordinate_dims_v<tf::range<Iterator1, N1>>;
    std::size_t axis = 0;
    auto lo = points[transversal[0]];
    auto hi = points[transversal[0]];
    for (const auto &id : transversal)
      for (std::size_t d = 0; d < Dims; ++d) {
        lo[d] = std::min(lo[d], points[id][d]);
        hi[d] = std::max(hi[d], points[id][d]);
      }
    for (std::size_t d = 1; d < Dims; ++d)
      if (hi[d] - lo[d] > hi[axis] - lo[axis])
        axis = d;
    std::sort(transversal.begin(), transversal.end(),
              [&points, axis](Index a, Index b) {
                if (points[a][axis] != points[b][axis])
                  return points[a][axis] < points[b][axis];
                return a < b;
              });
  }
  for (std::size_t i = 0; i + 1 < transversal.size(); i += 2)
    emit(transversal[i], transversal[i + 1], false);
}

/// Emit the chords of every (face, cut) group in one face's records.
///
/// The records of a face are sorted by cut, so a group is a run and the scan
/// that finds the runs IS the grouping — no offsets beside them.
template <typename Index, typename Iterator0, std::size_t N0,
          typename Iterator1, std::size_t N1, typename Emit>
auto for_each_cut_group(const tf::range<Iterator0, N0> &records,
                        Index face_size,
                        const tf::range<Iterator1, N1> &points,
                        const Emit &emit) -> void {
  for (std::size_t i = 0; i < records.size();) {
    std::size_t j = i + 1;
    while (j < records.size() && records[j].cut == records[i].cut)
      ++j;
    for_each_cut_chord(tf::slice(records, i, j), face_size, points, emit);
    i = j;
  }
}

} // namespace tf::intersect

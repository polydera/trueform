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
#include "../face_cuts.hpp"
#include "../loop_triangulations.hpp"

namespace tf::cut {

/// Emit the store's per-loop triangles into flat buffers, in loop
/// order. Each loop writes its (possibly aliased) range mapped through
/// its OWN tag — identity equality is what made a fold — and `rev`
/// flips the winding of folded opposing members.
template <typename Index, typename Int, typename MapVertex>
auto emit_stored_triangulations(const tf::face_cuts<Index, Int> &fc,
                                const loop_triangulations<Index, Int> &store,
                                const MapVertex &map_vertex,
                                tf::buffer<Index> &triangles,
                                tf::buffer<Index> &tag_labels,
                                tf::buffer<Index> &face_labels) -> void {
  auto descs = fc.descriptors();
  const auto n_loops = static_cast<Index>(descs.size());
  tf::buffer<Index> offsets;
  offsets.allocate(std::size_t(n_loops) + 1);
  offsets[0] = 0;
  for (Index l = 0; l < n_loops; ++l)
    offsets[std::size_t(l) + 1] =
        offsets[std::size_t(l)] + static_cast<Index>(store.loop_tris(l).size());
  const auto n_tris = offsets[std::size_t(n_loops)];
  triangles.allocate(std::size_t(n_tris) * 3);
  tag_labels.allocate(std::size_t(n_tris));
  face_labels.allocate(std::size_t(n_tris));
  tf::parallel_for_each(tf::make_sequence_range(n_loops), [&](Index l) {
    const auto &desc = descs[l];
    const bool rv = bool(store.rev[std::size_t(l)]);
    auto o = std::size_t(offsets[std::size_t(l)]);
    for (const auto &tr : store.loop_tris(l)) {
      triangles[3 * o] = map_vertex(desc.tag, tr[0]);
      triangles[3 * o + 1] = map_vertex(desc.tag, tr[rv ? 2 : 1]);
      triangles[3 * o + 2] = map_vertex(desc.tag, tr[rv ? 1 : 2]);
      tag_labels[o] = desc.tag;
      face_labels[o] = desc.object;
      ++o;
    }
  });
}

/// Origins-only overload for single-form partitions.
template <typename Index, typename Int, typename MapVertex>
auto emit_stored_triangulations(const tf::face_cuts<Index, Int> &fc,
                                const loop_triangulations<Index, Int> &store,
                                const MapVertex &map_vertex,
                                tf::buffer<Index> &triangles,
                                tf::buffer<Index> &face_origins) -> void {
  auto descs = fc.descriptors();
  const auto n_loops = static_cast<Index>(descs.size());
  tf::buffer<Index> offsets;
  offsets.allocate(std::size_t(n_loops) + 1);
  offsets[0] = 0;
  for (Index l = 0; l < n_loops; ++l)
    offsets[std::size_t(l) + 1] =
        offsets[std::size_t(l)] + static_cast<Index>(store.loop_tris(l).size());
  const auto n_tris = offsets[std::size_t(n_loops)];
  triangles.allocate(std::size_t(n_tris) * 3);
  face_origins.allocate(std::size_t(n_tris));
  tf::parallel_for_each(tf::make_sequence_range(n_loops), [&](Index l) {
    const auto &desc = descs[l];
    const bool rv = bool(store.rev[std::size_t(l)]);
    auto o = std::size_t(offsets[std::size_t(l)]);
    for (const auto &tr : store.loop_tris(l)) {
      triangles[3 * o] = map_vertex(desc.tag, tr[0]);
      triangles[3 * o + 1] = map_vertex(desc.tag, tr[rv ? 2 : 1]);
      triangles[3 * o + 2] = map_vertex(desc.tag, tr[rv ? 1 : 2]);
      face_origins[o] = desc.object;
      ++o;
    }
  });
}

} // namespace tf::cut

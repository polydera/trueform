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
#include "../../core/small_vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../compare_faces.hpp"
#include "../face_edge_neighbors.hpp"
#include <algorithm>
#include <atomic>
#include <iterator>

namespace tf::topology::domains {

/// @ingroup topology_components
/// @brief Resolve coincident duplicate-face stacks to their survivors.
///
/// Two faces stack when they share the same vertex-id cycle in either
/// winding — a purely topological test. Each stack keeps the face with
/// the smallest id (the same keep-one rule the csg path's coplanar
/// dedup applies through tags), so survivors of one stack all come
/// from the earliest source surface and are consistently wound with
/// it. Every face writes only its own slot: it collects its equal
/// neighbors through the shared first edge and takes the stack
/// minimum as its survivor.
///
/// `survivor_of[f] == f` marks a survivor (every non-stacked face is
/// its own survivor). Returns true when any stack exists.
template <typename Index, typename Faces, typename Fmem>
auto compute_face_stacks(const Faces &faces, const Fmem &fmem,
                         tf::buffer<Index> &survivor_of) -> bool {
  survivor_of.allocate(faces.size());
  std::atomic<bool> any{false};
  auto task_f = [&](Index face_id, tf::small_vector<Index, 6> &neighbors) {
    neighbors.clear();
    const auto &face = faces[face_id];
    tf::face_edge_neighbors(fmem, faces, face_id, Index(face[0]),
                            Index(face[1]), std::back_inserter(neighbors));
    Index survivor = face_id;
    const auto size = face.size();
    for (Index neighbor_id : neighbors) {
      if (neighbor_id == face_id || faces[neighbor_id].size() != size)
        continue;
      if (tf::compare_faces(face, faces[neighbor_id]) == 0)
        continue;
      survivor = std::min(survivor, neighbor_id);
      if (neighbor_id > face_id)
        any.store(true, std::memory_order_relaxed);
    }
    survivor_of[face_id] = survivor;
  };
  tf::parallel_for_each(tf::make_sequence_range(faces.size()), task_f,
                        tf::small_vector<Index, 6>{}, tf::checked);
  return any.load(std::memory_order_relaxed);
}

} // namespace tf::topology::domains

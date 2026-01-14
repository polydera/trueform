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

#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/faces.hpp"
#include "../core/none.hpp"
#include "../core/small_vector.hpp"
#include "../core/views/sequence_range.hpp"
#include "./are_faces_equal.hpp"
#include "./face_edge_neighbors.hpp"
#include "./face_membership_like.hpp"

namespace tf {

/// @ingroup topology_analysis
/// @brief Computes a boolean mask marking unique faces to keep.
///
/// Identifies duplicate faces in the mesh. Two faces are duplicates if they
/// have the same vertices in any cyclic order (either winding direction).
/// For each set of duplicate faces, the one with the smallest face index
/// is marked as unique (true), and all others are marked as duplicates (false).
///
/// @tparam Index The integer type for vertex indices (deduced if tf::none_t).
/// @tparam FacesPolicy The policy type for the faces range.
/// @tparam FmemPolicy The policy type for face membership.
/// @tparam MaskRange A range type supporting operator[] assignment.
/// @param faces The faces to check for duplicates.
/// @param fmem Pre-computed face membership structure.
/// @param mask Output mask where true = unique (keep), false = duplicate
/// (remove).
///
/// @note The mask must be pre-allocated to faces.size(). It is initialized
/// to true by this function.
/// @see tf::are_faces_equal() for the equality check used.
/// @see tf::face_membership for building the required connectivity structure.
template <typename Index = tf::none_t, typename FacesPolicy,
          typename FmemPolicy, typename MaskRange>
auto compute_unique_faces_mask(const tf::faces<FacesPolicy> &faces,
                               const tf::face_membership_like<FmemPolicy> &fmem,
                               MaskRange &mask) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(faces[0][0])>;
    return compute_unique_faces_mask<ActualIndex>(faces, fmem, mask);
  } else {
    tf::parallel_fill(mask, true);
    auto task_f = [&](Index face_id, tf::small_vector<Index, 6> &neighbors) {
      neighbors.clear();
      const auto &face = faces[face_id];

      // Extract all neighbors sharing first edge
      tf::face_edge_neighbors(fmem, faces, face_id, Index(face[0]),
                              Index(face[1]), std::back_inserter(neighbors));

      // Check if we're the smallest ID (cheap test)
      for (Index neighbor_id : neighbors) {
        if (neighbor_id < face_id)
          return; // Not smallest, skip
      }

      // We're smallest - mark duplicates as false (not unique)
      for (Index neighbor_id : neighbors) {
        if (tf::are_faces_equal(face, faces[neighbor_id])) {
          mask[neighbor_id] = false;
        }
      }
    };

    tf::parallel_for_each(tf::make_sequence_range(faces.size()), task_f,
                          tf::small_vector<Index, 6>{}, tf::checked);
  }
}
} // namespace tf

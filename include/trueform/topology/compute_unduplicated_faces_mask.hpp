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
/// @brief Computes a boolean mask marking faces with no duplicate.
///
/// Identifies duplicate faces in the mesh. Two faces are duplicates if they
/// have the same vertices in any cyclic order (either winding direction).
/// Unlike @ref tf::compute_unique_faces_mask, which keeps one representative
/// of each duplicate set, this drops EVERY face that has a duplicate: only
/// faces that appear exactly once are marked true.
///
/// This is the operation that cancels shared interfaces when merging meshes:
/// concatenate the boundaries of two adjacent regions and the faces of their
/// common interface appear twice (with opposite winding) and cancel, leaving
/// the boundary of the union.
///
/// @tparam Index The integer type for vertex indices (deduced if tf::none_t).
/// @tparam FacesPolicy The policy type for the faces range.
/// @tparam FmemPolicy The policy type for face membership.
/// @tparam MaskRange A range type supporting operator[] assignment.
/// @param faces The faces to check for duplicates.
/// @param fmem Pre-computed face membership structure.
/// @param mask Output mask where true = unique-appearance (keep), false =
/// has a duplicate (remove).
///
/// @note The mask must be pre-allocated to faces.size(). It is initialized
/// to true by this function.
/// @note Faces are matched only by shared vertices, so coincident-but-
/// separately-indexed copies must be merged first (e.g. with
/// @ref tf::cleaned using `remove_duplicate_primitives = false`).
/// @see tf::compute_unique_faces_mask() to instead keep one of each set.
/// @see tf::are_faces_equal() for the equality check used.
/// @see tf::face_membership for building the required connectivity structure.
template <typename Index = tf::none_t, typename FacesPolicy,
          typename FmemPolicy, typename MaskRange>
auto compute_unduplicated_faces_mask(
    const tf::faces<FacesPolicy> &faces,
    const tf::face_membership_like<FmemPolicy> &fmem, MaskRange &mask) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(faces[0][0])>;
    return compute_unduplicated_faces_mask<ActualIndex>(faces, fmem, mask);
  } else {
    tf::parallel_fill(mask, true);
    auto task_f = [&](Index face_id, tf::small_vector<Index, 6> &neighbors) {
      neighbors.clear();
      const auto &face = faces[face_id];

      // Extract all neighbors sharing first edge
      tf::face_edge_neighbors(fmem, faces, face_id, Index(face[0]),
                              Index(face[1]), std::back_inserter(neighbors));

      // Drop this face if any other face equals it
      auto size = face.size();
      for (Index neighbor_id : neighbors) {
        if (neighbor_id != face_id &&
            faces[neighbor_id].size() == size &&
            tf::are_faces_equal(face, faces[neighbor_id])) {
          mask[face_id] = false;
          return;
        }
      }
    };

    tf::parallel_for_each(tf::make_sequence_range(faces.size()), task_f,
                          tf::small_vector<Index, 6>{}, tf::checked);
  }
}
} // namespace tf

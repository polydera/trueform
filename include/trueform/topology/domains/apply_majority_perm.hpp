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
#include "../../core/views/zip.hpp"
#include <algorithm>

namespace tf::topology::domains {

/// @ingroup topology_components
/// @brief Phase D: reorder each valid NM edge's face block in place so
/// the per-position fragment-label sequence matches its set's majority
/// canonical radial perm.
///
/// Uses an in-place swap-walk: for each position `r`, scan forward for
/// the first slot whose label equals `majority_perm[r]` and swap it
/// into place. `O(K^2)` per edge, but `K` is the radial valence of the
/// edge — typically 2 to 4 — so this is cheap.
///
/// Reads:  `labels_view` (indirection into fragment_labels via
///         nm_edge_faces.data_buffer())
/// Writes: `nm_edge_faces.data_buffer()` in place
template <typename NMEdgeFaces, typename LabelsView, typename Index>
void apply_majority_perm(NMEdgeFaces &nm_edge_faces,
                         const LabelsView &labels_view,
                         const tf::buffer<char> &is_valid,
                         const tf::buffer<Index> &majority_rep) {
  tf::parallel_for_each(
      tf::zip(nm_edge_faces, labels_view, is_valid, majority_rep),
      [&](auto t) {
        auto &&[face_block, label_block, valid, rep] = t;
        if (!valid)
          return;
        auto maj_perm = labels_view[rep];
        Index K = Index(face_block.size());
        for (Index r = 0; r < K; ++r) {
          Index target = maj_perm[r];
          for (Index s = r; s < K; ++s) {
            if (label_block[s] == target) {
              if (s != r)
                std::swap(face_block[r], face_block[s]);
              break;
            }
          }
        }
      });
}

} // namespace tf::topology::domains

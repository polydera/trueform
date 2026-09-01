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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement {

/// Discover every uncut source face reached by accepted refinement splits.
///
/// `physical_rows` is sorted by `(tag, u, v, t)` and names undirected source
/// edges with `u < v`. The cut mask covers the complete standing descriptor
/// set. A membership hit is only a candidate: the face must contain the edge
/// as consecutive corners before its flat source-face id is emitted.
template <typename Index, typename PhysicalRows, typename CutMask,
          typename FaceOffsets, typename ApplyToForm>
auto discover_plane_refinement_entrants(const PhysicalRows &physical_rows,
                                        const CutMask &cut_mask,
                                        const FaceOffsets &face_offsets,
                                        const ApplyToForm &apply_to_form,
                                        tf::buffer<Index> &entrants) -> void {
  tf::buffer<std::array<Index, 3>> edges;
  edges.reserve(physical_rows.size());
  for (const auto &row : physical_rows) {
    const std::array<Index, 3> edge{row.tag, row.u, row.v};
    if (edges.size() == 0 || edges.back() != edge)
      edges.push_back(edge);
  }

  entrants.clear();
  tf::generic_generate(
      edges, entrants,
      [&](const std::array<Index, 3> &edge, tf::buffer<Index> &out) {
        const auto tag = edge[0];
        const auto u = edge[1];
        const auto v = edge[2];
        apply_to_form(tag, [&](const auto &form) {
          for (const auto face_id : form.face_membership()[u]) {
            if (cut_mask[std::size_t(tag)][std::size_t(face_id)])
              continue;
            const auto face = form.faces()[face_id];
            const auto size = face.size();
            for (std::size_t corner = 0; corner < size; ++corner) {
              const auto a = Index(face[corner]);
              const auto b = Index(face[(corner + 1) % size]);
              if ((a == u && b == v) || (a == v && b == u)) {
                out.push_back(face_offsets[std::size_t(tag)] + Index(face_id));
                break;
              }
            }
          }
        });
      },
      tf::checked);
  if (entrants.size() == 0)
    return;
  tbb::parallel_sort(entrants.begin(), entrants.end());
  entrants.erase_till_end(std::unique(entrants.begin(), entrants.end()));
}

} // namespace tf::arrangement

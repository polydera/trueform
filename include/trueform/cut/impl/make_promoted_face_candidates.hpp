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

#include "../../core/buffer.hpp"
#include "../../core/memory.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../face_regions.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int, typename ApplyToForm,
          typename OriginalEdgeSplit>
auto make_promoted_face_candidates(
    const tf::face_regions<Index, Int> &regions, Index n_tags,
    const ApplyToForm &apply_to_form,
    const tf::buffer<OriginalEdgeSplit> &original_edge_splits,
    const tf::buffer<std::array<Index, 3>> &promoted_steiners)
    -> tf::cut::detail::promoted_face_candidates<Index> {
  tf::cut::detail::promoted_face_candidates<Index> out;
  if (original_edge_splits.size() == 0)
    return out;

  auto descriptors = regions.descriptors();
  tf::core::std_vector<tf::buffer<char>> cut(
      static_cast<std::size_t>(n_tags));
  for (std::size_t tag = 0; tag < std::size_t(n_tags); ++tag) {
    apply_to_form(Index(tag), [&](const auto &form) {
      cut[tag].allocate(form.faces().size());
    });
    std::fill(cut[tag].begin(), cut[tag].end(), char(0));
  }
  for (Index region = 0; region < Index(descriptors.size()); ++region)
    if (descriptors[region].tag != Index(-1))
      cut[std::size_t(descriptors[region].tag)]
         [std::size_t(descriptors[region].object)] = 1;
  for (Index tag = 0; tag < n_tags; ++tag)
    for (auto face : regions.deleted(tag))
      cut[std::size_t(tag)][std::size_t(face)] = 1;

  for (const auto &split : original_edge_splits)
    apply_to_form(split.tag, [&](const auto &form) {
      for (auto face_id : form.face_membership()[split.u]) {
        if (cut[std::size_t(split.tag)][std::size_t(face_id)])
          continue;
        auto face = form.faces()[face_id];
        const Index size = Index(face.size());
        for (Index edge = 0; edge < size; ++edge) {
          const Index a = Index(face[std::size_t(edge)]);
          const Index b =
              Index(face[std::size_t((edge + 1) % size)]);
          if ((a == split.u && b == split.v) ||
              (a == split.v && b == split.u)) {
            out.faces.push_back({split.tag, Index(face_id)});
            break;
          }
        }
      }
    });
  std::sort(out.faces.begin(), out.faces.end());
  out.faces.erase_till_end(
      std::unique(out.faces.begin(), out.faces.end()));
  if (out.faces.size() == 0)
    return out;

  out.steiner_ranges.allocate(out.faces.size());
  std::size_t steiner = 0;
  for (std::size_t candidate = 0; candidate < out.faces.size();
       ++candidate) {
    while (steiner < promoted_steiners.size() &&
           (promoted_steiners[steiner][0] < out.faces[candidate][0] ||
            (promoted_steiners[steiner][0] ==
                 out.faces[candidate][0] &&
             promoted_steiners[steiner][1] <
                 out.faces[candidate][1])))
      ++steiner;
    const Index begin = Index(steiner);
    while (steiner < promoted_steiners.size() &&
           promoted_steiners[steiner][0] ==
               out.faces[candidate][0] &&
           promoted_steiners[steiner][1] ==
               out.faces[candidate][1])
      ++steiner;
    out.steiner_ranges[candidate] = {begin, Index(steiner)};
  }
  return out;
}

} // namespace tf::cut

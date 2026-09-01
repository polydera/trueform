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
#include "../../core/views/sequence_range.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>

namespace tf::arrangement {

/// THE PROMOTION'S COMPLETENESS, read off both tiers at the one barrier
/// that knows which source faces the world named.
///
/// A source face the world does not name is emitted from the source mesh
/// verbatim, so it can name only its own original vertices and its whole
/// original edges. The moment either tier moves one of those features,
/// every face holding it must enter the world — which is what the entrance
/// discovery does. So an uncut slot whose corner is a retired original, or
/// whose side is a root a split subdivided, is an entrance that did not
/// fire, and the output has a hole nothing refused to explain.
///
/// `face_of_slot` is the scatter that answers it: `-1` is exactly "the
/// world names no face here". The check is a debug one — its caller states
/// that — so the tables it sorts are built here and nothing keeps them.
template <typename Index, typename World, typename Arrangement,
          typename FaceCounts, typename VertexOffsets, typename ApplyToForm>
auto assert_promotion_is_complete(const World &world,
                                  const Arrangement &arrangement,
                                  const FaceCounts &face_counts,
                                  const VertexOffsets &vertex_offsets,
                                  const tf::buffer<Index> &face_of_slot,
                                  const ApplyToForm &apply_to_form) -> void {
  tf::buffer<Index> retired;
  for (const auto &row : world.merges())
    if (row[0] == Index(0))
      retired.push_back(row[1]);
  for (const auto &row : arrangement.merges())
    if (row[0] == Index(0))
      retired.push_back(row[1]);
  std::sort(retired.begin(), retired.end());
  retired.erase_till_end(std::unique(retired.begin(), retired.end()));

  tf::buffer<std::array<Index, 2>> split_sides;
  for (const auto root : world.split_roots()) {
    const auto &def = world.graph().canon_group(root)[0];
    if (def.point_tag_0 < 0 || def.point_tag_1 < 0)
      continue;
    split_sides.push_back(
        {vertex_offsets[std::size_t(def.point_tag_0)] + def.point_0,
         vertex_offsets[std::size_t(def.point_tag_1)] + def.point_1});
  }
  std::sort(split_sides.begin(), split_sides.end());

  tf::parallel_for_each(
      tf::make_sequence_range(face_of_slot.size()),
      [&](std::size_t slot) {
        if (face_of_slot[slot] != Index(-1))
          return;
        const auto tag =
            tf::exact::tag_of_flat_vertex(face_counts, Index(slot));
        const auto object = Index(slot) - face_counts[std::size_t(tag)];
        const auto base = vertex_offsets[std::size_t(tag)];
        apply_to_form(tag, [&](const auto &form) {
          const auto corners = form.faces()[object];
          const auto n = corners.size();
          for (std::size_t c = 0; c < n; ++c) {
            const auto from = base + Index(corners[c]);
            const auto to = base + Index(corners[(c + 1) % n]);
            assert(!std::binary_search(retired.begin(), retired.end(), from) &&
                   "an uncut source face names a retired original");
            const std::array<Index, 2> side{from < to ? from : to,
                                            from < to ? to : from};
            assert(!std::binary_search(split_sides.begin(), split_sides.end(),
                                       side) &&
                   "an uncut source face names a split original edge");
          }
        });
      },
      tf::checked);
}

} // namespace tf::arrangement

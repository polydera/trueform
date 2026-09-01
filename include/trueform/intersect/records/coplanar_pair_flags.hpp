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
#include "./tagged_intersection.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::intersect {

/// Order each pair's two (tag, object) halves and the table itself, so
/// the fact is one sorted key space.
template <typename Index>
auto normalize_coplanar_pairs(tf::buffer<std::array<Index, 4>> &pairs) -> void {
  for (auto &p : pairs)
    if (std::make_pair(p[0], p[1]) > std::make_pair(p[2], p[3]))
      p = {p[2], p[3], p[0], p[1]};
  std::sort(pairs.begin(), pairs.end());
  pairs.erase_till_end(std::unique(pairs.begin(), pairs.end()));
}

/// Stamp the pair-level coplanarity fact onto every record of a
/// coplanar pair. The fact belongs to the PAIR, not to an emission:
/// representative gating can route a contact through another pair's
/// call, and the duplicator clears the flag on copies it rewrites, so
/// the stamp is applied here, after both.
template <typename Index>
auto distribute_coplanar_flags(
    tf::buffer<tf::intersect::tagged_intersection<Index>> &records,
    const tf::buffer<Index> &offsets, const tf::buffer<Index> &partners,
    const tf::buffer<Index> &face_offsets) -> void {
  if (partners.size() == 0)
    return;
  tf::parallel_for_each(
      records,
      [&](tf::intersect::tagged_intersection<Index> &rec) {
        const auto face =
            std::size_t(face_offsets[std::size_t(rec.tag)] + rec.object);
        const auto other =
            face_offsets[std::size_t(rec.tag_other)] + rec.object_other;
        for (auto at = offsets[face]; at != offsets[face + 1]; ++at)
          if (partners[std::size_t(at)] == other) {
            rec.flags |= tf::intersect::coplanar_pair_flag;
            return;
          }
      },
      tf::checked);
}

} // namespace tf::intersect

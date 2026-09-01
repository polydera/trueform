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

#include <algorithm>
#include <cstddef>

namespace tf::arrangement {

/// CORE. A pooled carrier's emission stream, member-major.
///
/// Counts plus one prefix over the coverage give each member a disjoint run of
/// the triangles it emits, filled by one ascending pass, so the run IS the
/// emission — in the same order a walk over every triangle per member would
/// have produced, at the price of what is emitted. The member's output block
/// is therefore known before a triangle is written, and a duplicate emitted
/// first names its survivor's task-local position from the same table.
template <typename Index, typename Local, typename Labels>
auto gather_plane_member_triangles(Local &local, const Labels &labels,
                                   std::size_t n_tri, std::size_t n_members,
                                   Index base, bool record_survivors) -> void {
  local.member_offsets.allocate(n_members + 1);
  std::fill(local.member_offsets.begin(), local.member_offsets.end(), Index(0));
  for (std::size_t t = 0; t < n_tri; ++t) {
    const auto block = local.cover_range[std::size_t(labels[t])];
    for (auto at = block[0]; at != block[1]; ++at)
      ++local.member_offsets[std::size_t(local.cover[std::size_t(at)]) + 1];
  }
  for (std::size_t m = 1; m <= n_members; ++m)
    local.member_offsets[m] += local.member_offsets[m - 1];
  local.member_cursor.allocate(n_members);
  std::copy(local.member_offsets.begin(), local.member_offsets.end() - 1,
            local.member_cursor.begin());
  local.member_tri.allocate(std::size_t(local.member_offsets[n_members]));
  if (record_survivors)
    local.surv_pos.allocate(n_tri);
  for (std::size_t t = 0; t < n_tri; ++t) {
    const auto block = local.cover_range[std::size_t(labels[t])];
    for (auto at = block[0]; at != block[1]; ++at) {
      const auto mi = local.cover[std::size_t(at)];
      const auto slot = local.member_cursor[std::size_t(mi)]++;
      local.member_tri[std::size_t(slot)] = Index(t);
      if (record_survivors && local.surv_mi[std::size_t(labels[t])] == mi)
        local.surv_pos[t] = base + slot;
    }
  }
}

} // namespace tf::arrangement

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
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::intersect::graph {

/// The members of every plane once some leave and some join: per plane the
/// standing members a `retired` face (ascending) no longer is, in their own
/// order, then the joining faces `first_joining + k` the plane `joining[k]`
/// names, in `k` order. Counts plus one prefix, so every block is written
/// disjointly.
template <typename Index, typename MemberCount, typename Member>
auto state_live_plane_members(
    Index n_planes, Index n_standing_planes, const MemberCount &member_count,
    const Member &member, const tf::buffer<Index> &retired, Index first_joining,
    const tf::buffer<Index> &joining, tf::buffer<Index> &offsets,
    tf::buffer<Index> &members) -> void {
  const auto alive = [&](Index face) {
    return !std::binary_search(retired.begin(), retired.end(), face);
  };
  tf::buffer<std::array<Index, 2>> joined;
  joined.allocate(joining.size());
  for (std::size_t k = 0; k < joining.size(); ++k)
    joined[k] = {joining[k], Index(k)};
  std::sort(joined.begin(), joined.end());
  const auto joined_of = [&](Index p) {
    const auto begin = std::lower_bound(joined.begin(), joined.end(),
                                        std::array<Index, 2>{p, Index(0)});
    auto end = begin;
    while (end != joined.end() && (*end)[0] == p)
      ++end;
    return std::make_pair(begin, end);
  };
  offsets.allocate(std::size_t(n_planes) + 1);
  offsets[0] = Index(0);
  tf::parallel_for_each(
      tf::make_sequence_range(n_planes),
      [&](Index p) {
        Index count = 0;
        if (p < n_standing_planes)
          for (Index k = 0; k < member_count(p); ++k)
            count += alive(member(p, k)) ? Index(1) : Index(0);
        const auto range = joined_of(p);
        offsets[std::size_t(p) + 1] = count + Index(range.second - range.first);
      },
      tf::checked);
  for (std::size_t i = 1; i <= std::size_t(n_planes); ++i)
    offsets[i] += offsets[i - 1];
  members.allocate(std::size_t(offsets[std::size_t(n_planes)]));
  tf::parallel_for_each(
      tf::make_sequence_range(n_planes),
      [&](Index p) {
        auto write = std::size_t(offsets[std::size_t(p)]);
        if (p < n_standing_planes)
          for (Index k = 0; k < member_count(p); ++k) {
            const auto face = member(p, k);
            if (alive(face))
              members[write++] = face;
          }
        const auto range = joined_of(p);
        for (auto it = range.first; it != range.second; ++it)
          members[write++] = first_joining + (*it)[1];
      },
      tf::checked);
}

} // namespace tf::intersect::graph

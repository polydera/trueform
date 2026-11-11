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
#include "../../../../core/algorithm/parallel_for_each.hpp"
#include "../../../../core/algorithm/sequenced_generate.hpp"
#include "../../../../core/buffer.hpp"
#include "../../../../core/checked.hpp"
#include "../../../../core/hash_set.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "./land_tet_split_site.hpp"
#include "./tet_segment_recovery.hpp"
#include "./tet_site_is_corner.hpp"
#include <cstddef>
#include <tbb/parallel_sort.h>

namespace tf::topology::cdt::dt3 {

/// Settle where this round's fresh splits land: on a standing site, which
/// the split then shares — the one at its point, or a corner of the
/// constraints off its own segment in the point's cell of the recovery's
/// reach grid, which the point is (@ref tet_shares_reach_cell, @ref
/// tet_site_is_corner) — or on another fresh split. A split point lies exactly
/// on its own segment, so two that land on one point witness their segments
/// meeting there: the point is one identity, minted by the split with the
/// lowest ticket, and every other split landing on it waits a round, when its
/// row meets the minted site standing on it.
template <typename DT, typename Index, typename Coord>
auto settle_tet_split_sites(const DT &dt,
                            tet_segment_recovery<Index, Coord> &recovery)
    -> void {
  auto &splits = recovery.splits;
  struct local_t {
    tf::hash_set<Index> taken;
    tf::buffer<Index> flood;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(splits.size()),
      [&dt, &recovery, &splits](std::size_t i, local_t &local) {
        auto &split = splits[i];
        if (split.site != Index(-1))
          return;
        const auto &row = recovery.subsegments[std::size_t(split.subsegment)];
        local.taken.clear();
        split.site = land_tet_split_site(
            dt, split.position, row.hint, recovery.reach,
            [&recovery, &row](Index s) {
              return tet_site_is_corner(recovery, s, row.segment);
            },
            [&local](std::size_t c) {
              return local.taken.insert(Index(c)).second;
            },
            local.flood);
      },
      local_t{}, tf::checked(64));

  auto &records = recovery.candidates;
  records.clear();
  tf::sequenced_generate(
      tf::make_sequence_range(splits.size()), records,
      [&splits](std::size_t i, auto &out) {
        if (splits[i].site == Index(-1))
          out.push_back({splits[i].position, i});
      },
      tf::checked);
  tbb::parallel_sort(records.begin(), records.end(),
                     [](const auto &x, const auto &y) {
                       return x.position < y.position ||
                              (x.position == y.position && x.ticket < y.ticket);
                     });
  for (std::size_t i = 1; i < records.size(); ++i)
    if (records[i].position == records[i - 1].position) {
      auto &split = splits[records[i].ticket];
      recovery.revisits.push_back(split.subsegment);
      recovery.proposed[std::size_t(split.subsegment)] = Index(-1);
      split.subsegment = Index(-1);
    }
  std::size_t kept = 0;
  for (std::size_t i = 0; i < splits.size(); ++i)
    if (splits[i].subsegment != Index(-1))
      splits[kept++] = splits[i];
  splits.erase_till_end(splits.begin() + std::ptrdiff_t(kept));
}

} // namespace tf::topology::cdt::dt3

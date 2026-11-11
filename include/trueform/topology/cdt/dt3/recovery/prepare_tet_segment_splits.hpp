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
#include "../../../../core/buffer.hpp"
#include "../../../../core/checked.hpp"
#include "../../../../core/config/verify.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "../tet_site_position.hpp"
#include "./bucket_tet_records.hpp"
#include "./tet_segment_recovery.hpp"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// State what a split reads of the segments of `dt` (@ref
/// prepare_tet_segments): their ends, the first end of each one's first row
/// and the last end of its last, as runs by site with the acute ones marked,
/// no corner shelled and no ring placed, and no row proposed. A corner is acute
/// where two of its segments meet below sixty degrees; that is a choice of
/// where to protect, so it is read in double.
template <typename DT, typename Index, typename Coord>
auto prepare_tet_segment_splits(const DT &dt,
                                tet_segment_recovery<Index, Coord> &recovery)
    -> void {
  const auto &owner = dt.owner();
  const std::size_t n = recovery.status.size();
  const auto &rows = recovery.subsegments;
  recovery.ends.allocate(2 * n);
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&recovery, &rows](std::size_t s) {
        recovery.ends[2 * s] = {rows[std::size_t(recovery.first[s])].a,
                                Index(s), 0};
        recovery.ends[2 * s + 1] = {rows[std::size_t(recovery.last[s])].b,
                                    Index(s), 1};
      },
      tf::checked);
  {
    using end_t = std::decay_t<decltype(recovery.ends[0])>;
    bucket_tet_records(
        recovery.ends, dt.n_sites(), recovery.end_runs, recovery.end_scratch,
        recovery.bucket_counts, [](const end_t &end) { return end.site; },
        [](const end_t &x, const end_t &y) {
          return (x.segment) < (y.segment);
        });
  }

  const std::size_t m = recovery.ends.size();
  recovery.acute.allocate_and_initialize(m, char(0));
  recovery.shelled.allocate_and_initialize(m, char(0));
  recovery.rings.allocate_and_initialize(m, std::int64_t(-1));
  tf::parallel_for_each(
      tf::make_sequence_range(m),
      [&recovery, &owner, m](std::size_t i) {
        const auto &ends = recovery.ends;
        if (i && ends[i - 1].site == ends[i].site)
          return;
        std::size_t j = i;
        while (j < m && ends[j].site == ends[i].site)
          ++j;
        const auto direction = [&](std::size_t e, std::size_t axis) {
          const auto segment = std::size_t(ends[e].segment);
          const Index other =
              ends[e].end
                  ? recovery.subsegments[std::size_t(recovery.first[segment])].a
                  : recovery.subsegments[std::size_t(recovery.last[segment])].b;
          return double(tet_shallow_point(owner, other)[axis]) -
                 double(tet_shallow_point(owner, ends[e].site)[axis]);
        };
        bool acute = false;
        for (std::size_t x = i; x < j && !acute; ++x)
          for (std::size_t y = x + 1; y < j && !acute; ++y) {
            double xx = 0, yy = 0, xy = 0;
            for (std::size_t axis = 0; axis < 3; ++axis) {
              xx += direction(x, axis) * direction(x, axis);
              yy += direction(y, axis) * direction(y, axis);
              xy += direction(x, axis) * direction(y, axis);
            }
            acute = xy > 0 && 4 * xy * xy > xx * yy;
          }
        for (std::size_t x = i; x < j; ++x)
          recovery.acute[x] = char(acute);
      },
      tf::checked);
  recovery.proposed.allocate_and_initialize(rows.size(), Index(-1));
}

} // namespace tf::topology::cdt::dt3

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
#include "../../../../core/checked.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "../../../../exact/meta.hpp"
#include "../tet_site_position.hpp"
#include "./bucket_tet_records.hpp"
#include "./tet_segment_recovery.hpp"
#include "./tet_segment_status.hpp"
#include "./ticket_tet_subsegment.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// State segment `s` as row `s` on its own frame, its ends as runs by site
/// with the acute ones marked, and every live row due for the first round.
/// A corner is acute where two of its segments meet below sixty degrees;
/// that is a choice of where to protect, so it is read in double.
template <typename DT, typename Segments, typename Index, typename Coord>
auto prepare_tet_segments(const DT &dt, const Segments &segments, int depth,
                          tet_segment_recovery<Index, Coord> &recovery)
    -> void {
  using Int = typename DT::int_type;
  using T1 = typename tf::exact::meta<Int>::T1;
  const auto &owner = dt.owner();
  const std::size_t n = segments.size();
  const int ceiling = std::min(depth, tf::exact::meta<Int>::param_bits - 1);
  recovery.shell_splits = 0;
  recovery.status.allocate(n);
  recovery.rounds.allocate_and_initialize(n, Index(0));
  recovery.bits.allocate(n);
  recovery.first.allocate(n);
  recovery.last.allocate(n);
  recovery.subsegments.allocate(n);
  recovery.ends.allocate(2 * n);
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&recovery, &segments, &owner, ceiling](std::size_t s) {
        const Index a = Index(segments[s][0]), b = Index(segments[s][1]);
        int bits = ceiling;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          auto span = T1(tet_shallow_point(owner, b)[axis]) -
                      T1(tet_shallow_point(owner, a)[axis]);
          int zeros = 0;
          while (span != T1(0) && (span & T1(1)) == T1(0) && zeros < bits) {
            span = span / T1(2);
            ++zeros;
          }
          if (span != T1(0))
            bits = std::min(bits, zeros);
        }
        recovery.status[s] = a == b ? tet_segment_status::degenerate
                                    : tet_segment_status::pending;
        recovery.bits[s] = bits;
        recovery.first[s] = recovery.last[s] = Index(s);
        recovery.subsegments[s] = {Index(s), a, b, 0, std::int64_t(1) << bits,
                                   Index(0), 0};
        recovery.ends[2 * s] = {a, Index(s), 0};
        recovery.ends[2 * s + 1] = {b, Index(s), 1};
      },
      tf::checked);
  recovery.row_heads.allocate_and_initialize(dt.n_sites(), Index(-1));
  recovery.row_tickets.clear();
  recovery.revisits.clear();
  for (std::size_t s = 0; s < n; ++s) {
    ticket_tet_subsegment(recovery, recovery.subsegments[s].a, Index(s));
    ticket_tet_subsegment(recovery, recovery.subsegments[s].b, Index(s));
  }
  {
    using end_t = std::decay_t<decltype(recovery.ends[0])>;
    tf::buffer<end_t> scratch;
    tf::buffer<Index> runs;
    bucket_tet_records(
        recovery.ends, dt.n_sites(), runs, scratch,
        [](const end_t &end) { return end.site; },
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
          const auto &row = recovery.subsegments[std::size_t(ends[e].segment)];
          const Index other = ends[e].end ? row.a : row.b;
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

  recovery.dirty.clear();
  tf::sequenced_generate(
      tf::make_sequence_range(n), recovery.dirty,
      [&recovery](std::size_t s, auto &out) {
        if (recovery.status[s] == tet_segment_status::pending)
          out.push_back(Index(s));
      },
      tf::checked);
  recovery.proposed.allocate_and_initialize(n, Index(-1));
  recovery.first_minted = dt.n_sites();
  recovery.minted.clear();
  recovery.placed.clear();
  recovery.placed_from = 0;
  recovery.stamps.allocate_and_initialize(dt.n_sites(), std::uint32_t(0));
  recovery.stamp = 0;
}

} // namespace tf::topology::cdt::dt3

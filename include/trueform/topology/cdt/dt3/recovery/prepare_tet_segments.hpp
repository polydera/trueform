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
#include "../../../../core/config/verify.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "../../../../exact/meta.hpp"
#include "../tet_site_position.hpp"
#include "./tet_segment_recovery.hpp"
#include "./tet_segment_status.hpp"
#include "./ticket_tet_subsegment.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// State every segment of `segments` from `standing` on, the ones
/// `recovery` already holds before it, as a row on its own frame, ticketed at
/// its lower end, its first and last row, and every new pending row due for
/// the next round, after the pending rows `revisits` names; the segments
/// `recovery` already holds keep their rows, chains and statuses, so a
/// `standing` of zero states every segment afresh, segment `s` as row `s`.
/// What a split reads of the segments is stated when the first split is due
/// (@ref prepare_tet_segment_splits).
template <typename DT, typename Segments, typename Index, typename Coord>
auto prepare_tet_segments(const DT &dt, const Segments &segments,
                          std::size_t standing, int depth,
                          tet_segment_recovery<Index, Coord> &recovery)
    -> void {
  using Int = typename DT::int_type;
  using T1 = typename tf::exact::meta<Int>::T1;
  const auto &owner = dt.owner();
  if constexpr (tf::core::verification)
    tf::core::verify(!standing || standing == recovery.status.size(),
                     "a diff on the segments keeps the ones the recovery "
                     "holds");
  if (!standing) {
    recovery.steiner.clear();
    recovery.status.clear();
    recovery.rounds.clear();
    recovery.bits.clear();
    recovery.first.clear();
    recovery.last.clear();
    recovery.subsegments.clear();
  }
  const std::size_t n = segments.size();
  const std::size_t base = recovery.subsegments.size();
  const int ceiling = std::min(depth, tf::exact::meta<Int>::param_bits - 1);
  if (!standing) {
    recovery.shell_splits = 0;
    recovery.flip_recovered = 0;
    recovery.search_recovered = 0;
    recovery.split_rows = 0;
    recovery.crossing_rows = 0;
    recovery.round = 0;
    recovery.flips.hull_barred = 0;
    recovery.row_heads.clear();
  }
  recovery.status.reallocate(n);
  recovery.rounds.reallocate_and_initialize(n, Index(0));
  recovery.bits.reallocate(n);
  recovery.first.reallocate(n);
  recovery.last.reallocate(n);
  recovery.subsegments.reallocate(base + (n - standing));
  recovery.row_links.reallocate(base + (n - standing));
  tf::parallel_for_each(
      tf::make_sequence_range(standing, n),
      [&recovery, &segments, &owner, ceiling, base, standing](std::size_t s) {
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
        const Index row = Index(base + (s - standing));
        recovery.status[s] = a == b ? tet_segment_status::degenerate
                                    : tet_segment_status::pending;
        recovery.bits[s] = bits;
        recovery.first[s] = recovery.last[s] = row;
        recovery.subsegments[std::size_t(row)] = {
            Index(s), a, b, Index(-1), 0, std::int64_t(1) << bits, Index(0), 0};
      },
      tf::checked);
  recovery.row_heads.reallocate_and_initialize(dt.n_sites(), Index(-1));
  for (std::size_t s = standing; s < n; ++s)
    ticket_tet_subsegment(recovery, Index(base + (s - standing)));
  recovery.dirty.clear();
  for (auto k : recovery.revisits)
    if (recovery.status[std::size_t(
            recovery.subsegments[std::size_t(k)].segment)] ==
        tet_segment_status::pending)
      recovery.dirty.push_back(k);
  recovery.revisits.clear();
  tf::sequenced_generate(
      tf::make_sequence_range(standing, n), recovery.dirty,
      [&recovery, base, standing](std::size_t s, auto &out) {
        if (recovery.status[s] == tet_segment_status::pending)
          out.push_back(Index(base + (s - standing)));
      },
      tf::checked);
  recovery.first_minted = dt.n_sites();
  recovery.minted.clear();
  recovery.proposed.allocate_and_initialize(recovery.subsegments.size(),
                                            Index(-1));
}

} // namespace tf::topology::cdt::dt3

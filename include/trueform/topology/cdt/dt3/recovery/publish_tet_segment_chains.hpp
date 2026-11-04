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
#include "../../../../core/algorithm/generate_offset_blocks.hpp"
#include "../../../../core/algorithm/parallel_iota.hpp"
#include "../../../../core/checked.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "./bucket_tet_records.hpp"
#include "./tet_segment_recovery.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// Publish every segment's chain, block `s` for segment `s`: its rows in
/// `order`, sorted by segment and first end, walked from the one row whose
/// first end is the segment's own, each vertex with its numerator lifted
/// onto the 2^depth grid. The rows themselves stay where their tickets name
/// them.
template <typename Index, typename Coord>
auto publish_tet_segment_chains(int depth,
                                tet_segment_recovery<Index, Coord> &recovery)
    -> void {
  const auto &rows = recovery.subsegments;
  auto &order = recovery.order;
  order.allocate(rows.size());
  tf::parallel_iota(order, Index(0));
  {
    tf::buffer<Index> scratch;
    tf::buffer<Index> runs;
    bucket_tet_records(
        order, recovery.status.size(), runs, scratch,
        [&rows](Index k) { return rows[std::size_t(k)].segment; },
        [&rows](Index x, Index y) {
          return rows[std::size_t(x)].a < rows[std::size_t(y)].a;
        });
  }
  recovery.chains.clear();
  tf::generate_offset_blocks(
      tf::make_sequence_range(recovery.status.size()),
      recovery.chains.offsets_buffer(), recovery.chains.data_buffer(),
      [&recovery, &rows, &order, depth](std::size_t s, auto &out) {
        const int lift = depth - recovery.bits[s];
        const auto from = std::lower_bound(
            order.begin(), order.end(), s,
            [&rows](Index k, std::size_t segment) {
              return std::size_t(rows[std::size_t(k)].segment) < segment;
            });
        auto to = from;
        while (to != order.end() &&
               std::size_t(rows[std::size_t(*to)].segment) == s)
          ++to;
        auto at = from;
        while (rows[std::size_t(*at)].n0 != 0)
          ++at;
        const auto count = std::size_t(to - from);
        for (std::size_t step = 0; step < count; ++step) {
          const auto &row = rows[std::size_t(*at)];
          out.push_back({row.a, row.n0 < 0 ? row.n0 : row.n0 << lift});
          if (step + 1 < count)
            at = std::lower_bound(from, to, row.b,
                                  [&rows](Index k, Index value) {
                                    return rows[std::size_t(k)].a < value;
                                  });
        }
        out.push_back({rows[std::size_t(*at)].b, std::int64_t(1) << depth});
      },
      tf::checked);
}

} // namespace tf::topology::cdt::dt3

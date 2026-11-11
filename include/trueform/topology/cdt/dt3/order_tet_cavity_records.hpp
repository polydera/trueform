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
#include "../../../core/config/verify.hpp"
#include "./tet_cavity_links.hpp"
#include "./tet_execution_tuning.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// States a boundary's edge records, one per facet edge as its lower corner,
/// its upper corner and its ticket, ordered by the lower corner, stably: the
/// digits' counts are taken in `record_counts` as the records are stated, and
/// each digit the highest corner slot holds is one scatter through
/// `record_scratch`. A corner stamps inside `local_of_site`, so the digits of
/// its slots are the lanes counted.
template <typename Index>
auto order_tet_cavity_records(tet_cavity_links<Index> &links) -> void {
  constexpr std::size_t bits = tet_execution_tuning::cavity_record_bits;
  constexpr std::size_t buckets = std::size_t(1) << bits;
  constexpr std::size_t digits = (8 * sizeof(Index) + bits - 1) / bits;
  const std::size_t slots = links.local_of_site.size();
  std::size_t lanes = 1;
  while (lanes < digits && ((slots - 1) >> (bits * lanes)) != 0)
    ++lanes;
  auto &ends = links.record_counts;
  if (ends.size() < lanes * buckets)
    ends.allocate(lanes * buckets);
  std::fill(ends.data(), ends.data() + lanes * buckets, std::uint32_t(0));
  auto &records = links.records;
  records.allocate(3 * links.n_facets);
  std::size_t highest = 0;
  for (std::size_t i = 0; i < links.n_facets; ++i) {
    const auto &facet = links.facets[i];
    for (std::size_t e = 0; e < 3; ++e) {
      const Index a = facet[(e + 1) % 3], b = facet[(e + 2) % 3];
      const Index lo = a < b ? a : b;
      records[3 * i + e] = {lo, a < b ? b : a, Index(3 * i + e)};
      const std::size_t key = tet_cavity_stamp_slot(lo);
      highest = highest < key ? key : highest;
      for (std::size_t lane = 0; lane < lanes; ++lane)
        ++ends[lane * buckets + ((key >> (bits * lane)) & (buckets - 1))];
    }
  }
  if constexpr (tf::core::verification)
    tf::core::verify(highest < slots,
                     "every boundary corner stamps inside its lane");
  auto &scratch = links.record_scratch;
  scratch.allocate(records.size());
  for (std::size_t lane = 0; lane < lanes && (highest >> (bits * lane)) != 0;
       ++lane) {
    auto end = ends.data() + lane * buckets;
    std::uint32_t before = 0;
    for (std::size_t bucket = 0; bucket < buckets; ++bucket) {
      const std::uint32_t count = end[bucket];
      end[bucket] = before;
      before += count;
    }
    for (const auto &record : records)
      scratch[end[(tet_cavity_stamp_slot(record[0]) >> (bits * lane)) &
                  (buckets - 1)]++] = record;
    std::swap(records, scratch);
  }
}

} // namespace tf::topology::cdt::dt3

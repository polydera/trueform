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
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/grain.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./prepare_tet_claim_band.hpp"
#include "./redistribute_tet_claim_rows.hpp"
#include "./run_tet_claim_stream.hpp"
#include "./tet_claim_scratch.hpp"
#include "./tet_claim_workspace.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

template <typename Owner, typename Index>
auto insert_tet_claim_band(Owner &owner, tet_claim_workspace<Index> &work,
                           std::size_t begin, std::size_t end) -> void {
  prepare_tet_claim_band(owner, work, begin, end);
  ++owner._stats.parallel_bands;
  for (;;) {
    tf::parallel_for_each(
        tf::make_sequence_range(work.streams.size()),
        [&](std::size_t stream, tet_claim_scratch<Index> &scratch) {
          run_tet_claim_stream(owner, work, stream, scratch);
        },
        tet_claim_scratch<Index>{}, tf::grain(1));
    if (!redistribute_tet_claim_rows(work))
      break;
    ++owner._stats.pool_restarts;
  }
  auto inserted = end - begin;
  for (const auto &stream : work.streams) {
    inserted -= stream.end - stream.begin;
    for (auto i = stream.begin; i < stream.end; ++i)
      work.tail.push_back(owner._order[i]);
  }
  owner._stats.parallel_insertions += inserted;
  owner._stats.insertions += inserted;
}

} // namespace tf::topology::cdt::dt3

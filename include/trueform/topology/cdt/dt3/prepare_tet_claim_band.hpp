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
#include "./locate_claimed_tet.hpp"
#include "./release_tet_cells.hpp"
#include "./tet_claim_scratch.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include <atomic>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

template <typename Owner, typename Index>
auto prepare_tet_claim_band(Owner &owner, tet_claim_workspace<Index> &work,
                            std::size_t begin, std::size_t end) -> void {
  Index seed = -1;
  for (std::size_t row = 0; row < work.ownership.size(); ++row)
    if (work.ownership[row].load(std::memory_order_relaxed) == tet_unowned) {
      seed = Index(row);
      break;
    }
  if constexpr (tf::core::verification)
    tf::core::verify(seed >= 0, "a band opens on a row no pool holds");
  // A band opens with no stream running, so stream 0's lists are free.
  auto &scratch = work.scratch[0].value;
  const auto locate_hint = [&](Index site, Index from) {
    scratch.held.clear();
    const auto result = locate_claimed_tet(owner, work, 0, site, from, scratch);
    release_tet_cells(work, 0, scratch.held, 0);
    return result;
  };
  const auto streams = work.streams.size();
  for (std::size_t i = 0; i < streams; ++i) {
    auto &stream = work.streams[i];
    stream.begin = begin + (end - begin) * i / streams;
    stream.end = begin + (end - begin) * (i + 1) / streams;
    stream.stopping = 0;
    if (stream.begin < stream.end) {
      seed = locate_hint(owner._order[stream.begin], seed);
      stream.hint_begin = seed;
      seed = locate_hint(owner._order[stream.end - 1], seed);
      stream.hint_end = seed;
    }
  }
}

} // namespace tf::topology::cdt::dt3

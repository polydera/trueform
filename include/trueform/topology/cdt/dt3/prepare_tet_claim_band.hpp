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
#include "./locate_claimed_tet.hpp"
#include "./order_tet_site_band.hpp"
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
  order_tet_site_band(owner, begin, end);
  Index seed = 0;
  while (work.ownership[std::size_t(seed)].load(std::memory_order_relaxed) !=
         tet_unowned)
    ++seed;
  tet_claim_scratch<Index> scratch{};
  const auto locate_hint = [&](Index site, Index from) {
    scratch.held.clear();
    const auto result = locate_claimed_tet(owner, work, 0, site, from, scratch);
    release_tet_cells(work, 0, scratch.held);
    return result;
  };
  const auto streams = work.streams.size();
  for (std::size_t i = 0; i < streams; ++i) {
    auto &stream = work.streams[i];
    stream.begin = begin + (end - begin) * i / streams;
    stream.end = begin + (end - begin) * (i + 1) / streams;
    if (stream.begin < stream.end) {
      seed = locate_hint(owner._order[stream.begin], seed);
      stream.hint_begin = seed;
      seed = locate_hint(owner._order[stream.end - 1], seed);
      stream.hint_end = seed;
    }
  }
}

} // namespace tf::topology::cdt::dt3

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
#include "./tet_claim_workspace.hpp"
#include "./tet_execution_tuning.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// At the barrier, move free rows from finished streams to blocked ones;
/// whether any moved.
template <typename Index>
inline auto redistribute_tet_claim_rows(tet_claim_workspace<Index> &work)
    -> bool {
  using Tuning = tet_execution_tuning;
  bool moved = false;
  for (auto &target : work.streams) {
    if (target.begin == target.end)
      continue;
    auto wanted = Tuning::borrowed_rows_per_site * (target.end - target.begin) +
                  Tuning::borrowed_rows_floor;
    for (auto &source : work.streams) {
      if (source.begin != source.end || source.free_count == 0)
        continue;
      if (source.fresh_begin < source.fresh_end) {
        work.free_next[source.fresh_end - 1] = source.free_head;
        source.free_head = Index(source.fresh_begin);
        source.fresh_begin = source.fresh_end;
      }
      const auto count = std::min(wanted, source.free_count);
      const auto first = source.free_head;
      auto last = first;
      for (std::size_t i = 1; i < count; ++i)
        last = work.free_next[std::size_t(last)];
      source.free_head = work.free_next[std::size_t(last)];
      work.free_next[std::size_t(last)] = target.free_head;
      target.free_head = first;
      source.free_count -= count;
      target.free_count += count;
      wanted -= count;
      moved = true;
      if (wanted == 0)
        break;
    }
  }
  return moved;
}

} // namespace tf::topology::cdt::dt3

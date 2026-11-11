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
#include "./tet_claim_rows_asked.hpp"
#include "./tet_claim_stream.hpp"
#include "./tet_claim_workspace.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// At a barrier, give every stream holding fewer rows than it asks for at
/// `density` (@ref tet_claim_rows_asked) what it lacks, from the rows each
/// other stream holds beyond its own ask; whether any moved.
template <typename Index>
inline auto redistribute_tet_claim_rows(tet_claim_workspace<Index> &work,
                                        double density) -> bool {
  bool moved = false;
  for (auto &target : work.streams) {
    const auto asked = tet_claim_rows_asked(target, density);
    if (target.free_count >= asked)
      continue;
    auto wanted = asked - target.free_count;
    for (auto &source : work.streams) {
      const auto kept = tet_claim_rows_asked(source, density);
      if (source.free_count <= kept)
        continue;
      if (source.fresh_begin < source.fresh_end) {
        work.free_next[source.fresh_end - 1] = source.free_head;
        source.free_head = Index(source.fresh_begin);
        source.fresh_begin = source.fresh_end;
      }
      const auto count = std::min(wanted, source.free_count - kept);
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

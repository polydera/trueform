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
#include "./insert_claimed_tet_site.hpp"
#include "./tet_claim_scratch.hpp"
#include "./tet_claim_workspace.hpp"
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// One pass of stream `worker` over its slice `[begin, end)` of the order,
/// taken from either end. The stream owns its slice, its free rows and, for
/// one attempt, the cells that attempt claimed. An attempt that meets a cell
/// another stream holds releases every claim it made; the site stays in the
/// slice, collided, and the stream turns to its other end, neither waiting
/// for the holder nor trying the site again in this pass. A pass therefore
/// ends within one attempt per site whatever the other streams do, or
/// whether they run at all, and the slice it leaves holds its collided sites
/// and those its rows could not take. The Delaunay complex of the sites under
/// the name tie rule is unique, so which stream inserts a site, and in which
/// round, moves only the work; the build publishes the complex canonically.
template <typename Owner, typename Index>
auto run_tet_claim_stream(Owner &owner, tet_claim_workspace<Index> &work,
                          std::size_t worker, tet_claim_scratch<Index> &scratch)
    -> void {
  auto &stream = work.streams[worker];
  Index hints[2]{stream.hint_begin, stream.hint_end};
  auto front = stream.begin, back = stream.end;
  bool reverse = false;
  stream.stopping = 0;
  while (front < back) {
    const auto pos = reverse ? back - 1 : front;
    const auto result =
        insert_claimed_tet_site(owner, work, worker, owner._order[pos],
                                hints[reverse ? 1 : 0], scratch);
    if (result == tet_claim_result::capacity) {
      stream.stopping = scratch.boundary.size() - scratch.cavity.size();
      break;
    }
    if (result == tet_claim_result::inserted) {
      if (reverse)
        std::swap(owner._order[--back], owner._order[--stream.end]);
      else
        std::swap(owner._order[front++], owner._order[stream.begin++]);
      continue;
    }
    if (reverse)
      --back;
    else
      ++front;
    reverse = !reverse;
  }
  stream.hint_begin = hints[0];
  stream.hint_end = hints[1];
}

} // namespace tf::topology::cdt::dt3

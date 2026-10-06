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
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>

namespace tf::topology::cdt::dt3 {

template <typename Owner, typename Index>
auto run_tet_claim_stream(Owner &owner, tet_claim_workspace<Index> &work,
                          std::size_t worker, tet_claim_scratch<Index> &scratch)
    -> void {
  auto &stream = work.streams[worker];
  Index hints[2]{stream.hint_begin, stream.hint_end};
  bool reverse = false;
  while (stream.begin < stream.end) {
    const auto pos = reverse ? stream.end - 1 : stream.begin;
    auto &hint = hints[reverse ? 1 : 0];
    const auto result = insert_claimed_tet_site(
        owner, work, worker, owner._order[pos], hint, scratch);
    if (result == tet_claim_result::capacity)
      break;
    if (result == tet_claim_result::inserted) {
      if (reverse)
        --stream.end;
      else
        ++stream.begin;
    } else {
      const auto other = scratch.interfering;
      if (other >= 0 && worker < std::size_t(other)) {
        if (scratch.interfering_cell >= 0) {
          const auto &word =
              work.ownership[std::size_t(scratch.interfering_cell)];
          for (;;) {
            const auto held = word.load(std::memory_order_relaxed);
            if ((held & tet_worker_mask) != std::uint32_t(other) ||
                (held & tet_pooled))
              break;
            std::this_thread::yield();
          }
        }
      } else
        reverse = !reverse;
    }
  }
  stream.hint_begin = hints[0];
  stream.hint_end = hints[1];
}

} // namespace tf::topology::cdt::dt3

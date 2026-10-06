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
#include "../../../core/views/sequence_range.hpp"
#include "./tet_claim_stream.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

template <typename Owner, typename Index>
auto prepare_tet_claim_pool(Owner &owner, tet_claim_workspace<Index> &work,
                            std::size_t workers, std::size_t count) -> void {
  const std::size_t live = std::size_t(owner._corners.size());
  owner._corners.reallocate(count);
  owner._neighbors.reallocate(count);
  owner._states.reallocate(count);
  work.ownership.allocate_and_initialize(count, tet_unowned);
  work.free_next.allocate(count);
  work.streams.allocate_and_initialize(workers, tet_claim_stream<Index>{});
  tf::parallel_for_each(
      tf::make_sequence_range(workers), [&](std::size_t worker) {
        const auto begin = live + (count - live) * worker / workers,
                   end = live + (count - live) * (worker + 1) / workers;
        auto &stream = work.streams[worker];
        stream.fresh_begin = begin;
        stream.fresh_end = end;
        stream.free_count = end - begin;
        for (auto i = begin; i < end; ++i) {
          work.free_next[i] = i + 1 < end ? Index(i + 1) : -1;
          work.ownership[i].store(std::uint32_t(worker) | tet_pooled,
                                  std::memory_order_relaxed);
        }
      });
}

} // namespace tf::topology::cdt::dt3

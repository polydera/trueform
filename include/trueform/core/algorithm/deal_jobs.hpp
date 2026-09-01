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

#include "../buffer.hpp"
#include "../checked.hpp"
#include "./block_reduce_sequenced_aggregate.hpp"
#include "./parallel_transform.hpp"
#include "tbb/parallel_sort.h"

#include <cstddef>

namespace tf {
namespace core {

/// One job and the count that predicts its cost.
template <typename Index> struct job_cost {
  Index cost;
  Index job;
};

} // namespace core

/// @ingroup core_algorithms
/// @brief Deal the jobs so the expensive ones land in DIFFERENT chunks of the
///        pass that runs them.
///
/// Heaviest predicted cost first, ties by job id, then round-robin across the
/// chunk stride. A job's cost can span orders of magnitude and the heaviest
/// jobs can be adjacent ids, so equal-count contiguous chunking collects them
/// into one chunk and that chunk becomes the whole pass.
///
/// The order is a schedule, not a product: every carrier the pass writes is
/// addressed by job id, so the same deal is free to differ with the machine.
template <typename Index, typename Jobs, typename CostOf>
auto deal_jobs(const Jobs &jobs, const CostOf &cost_of,
               tf::buffer<Index> &order) -> void {
  const auto n = jobs.size();
  tf::buffer<core::job_cost<Index>> ranked;
  ranked.allocate(n);
  tf::parallel_transform(
      jobs, ranked,
      [&](Index job) { return core::job_cost<Index>{cost_of(job), job}; },
      tf::checked);
  tbb::parallel_sort(
      ranked.begin(), ranked.end(),
      [](const core::job_cost<Index> &x, const core::job_cost<Index> &y) {
        if (x.cost != y.cost)
          return y.cost < x.cost;
        return x.job < y.job;
      });
  order.allocate(n);
  const auto step = tf::core::blocked_sequenced_chunk_stride(n);
  std::size_t rank = 0;
  for (std::size_t slot = 0; slot < step; ++slot)
    for (std::size_t at = slot; at < n; at += step)
      order[at] = ranked[rank++].job;
}

} // namespace tf

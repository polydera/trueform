/**
 * @file worker_ladder.hpp
 * @brief The worker counts a test runs a schedule at
 *
 * A test that states a product is independent of the worker count states
 * it at the machine's concurrency against one thread, every count it runs
 * taken from here and never past the machine.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <tbb/task_arena.h>
#include <vector>

namespace tf::test {

/// The concurrency of the arena a test case starts in: the machine's.
inline auto machine_workers() -> std::size_t {
  return std::size_t(tbb::this_task_arena::max_concurrency());
}

/// `count`, never past @ref machine_workers.
inline auto capped_workers(std::size_t count) -> std::size_t {
  return std::min(count, machine_workers());
}

/// `counts` in order, each capped and kept once.
inline auto worker_ladder(std::initializer_list<std::size_t> counts)
    -> std::vector<std::size_t> {
  std::vector<std::size_t> ladder;
  for (const std::size_t count : counts) {
    const std::size_t capped = capped_workers(count);
    if (std::find(ladder.begin(), ladder.end(), capped) == ladder.end())
      ladder.push_back(capped);
  }
  return ladder;
}

} // namespace tf::test

/**
 * @file test_parallel_workers.cpp
 * @brief Tests for tf::core::parallel_workers, the workers an arena will run.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "worker_ladder.hpp"
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/config/parallel_workers.hpp>

#include <tbb/global_control.h>
#include <tbb/task_arena.h>

#include <cstddef>

TEST_CASE("parallel_workers: the arena's concurrency, bounded by the active "
          "global limit",
          "[core][parallel]") {
  const std::size_t machine = tf::test::machine_workers();
  REQUIRE(tf::core::parallel_workers() == machine);
  {
    const tbb::global_control limit(
        tbb::global_control::max_allowed_parallelism, 1);
    REQUIRE(std::size_t(tbb::this_task_arena::max_concurrency()) == machine);
    REQUIRE(tf::core::parallel_workers() == 1);
  }
  REQUIRE(tf::core::parallel_workers() == machine);

  const std::size_t four = tf::test::capped_workers(4);
  tbb::task_arena arena_of_four(static_cast<int>(four));
  REQUIRE(arena_of_four.execute([] { return tf::core::parallel_workers(); }) ==
          four);

  const std::size_t eight = tf::test::capped_workers(8);
  const std::size_t two = tf::test::capped_workers(2);
  const tbb::global_control limit(tbb::global_control::max_allowed_parallelism,
                                  two);
  tbb::task_arena arena_of_eight(static_cast<int>(eight));
  REQUIRE(arena_of_eight.execute([] {
    return std::size_t(tbb::this_task_arena::max_concurrency());
  }) == eight);
  REQUIRE(arena_of_eight.execute([] { return tf::core::parallel_workers(); }) ==
          two);
}

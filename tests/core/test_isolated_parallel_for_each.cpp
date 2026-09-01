/**
 * @file test_isolated_parallel_for_each.cpp
 * @brief Tests for cancellation-isolated tf::parallel_for_each.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/algorithm.hpp>

#include <tbb/task_group.h>

#include <array>
#include <atomic>
#include <cstddef>

TEST_CASE("parallel_for_each isolated visits every value after cancellation",
          "[core][algorithm][parallel]") {
  constexpr std::size_t size = 4096;
  std::array<std::size_t, size> values{};
  std::array<std::atomic<int>, size> visits{};
  for (std::size_t i = 0; i < size; ++i) {
    values[i] = i;
    visits[i].store(0, std::memory_order_relaxed);
  }

  tbb::task_group_context context;
  tbb::task_group group(context);
  bool called = false;
  group.run([&] {
    called = true;
    context.cancel_group_execution();
    tf::parallel_for_each(
        values,
        [&](std::size_t value) {
          visits[value].fetch_add(1, std::memory_order_relaxed);
        },
        tf::isolated);
  });
  group.wait();

  REQUIRE(called);
  for (const auto &visit : visits)
    REQUIRE(visit.load(std::memory_order_relaxed) == 1);
}

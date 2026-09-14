/**
 * @file test_generate_offset_blocks.cpp
 * @brief The offset-block generation's own laws: the empty structure, and the
 *        checked entry every primitive of the vocabulary takes.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/algorithm/generate_offset_blocks.hpp>
#include <trueform/core/algorithm/parallel_for.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/checked.hpp>
#include <trueform/core/views/sequence_range.hpp>

#include <atomic>
#include <cstddef>

/// AN OFFSET-BLOCK STRUCTURE WITH NO BLOCKS HAS EMPTY OFFSETS. A reused buffer
/// must not keep the fences of the build before it, so a generation over
/// nothing states nothing rather than leaving what was there.
TEST_CASE("generate_offset_blocks states an empty structure",
          "[core][algorithm][offset-blocks]") {
  tf::buffer<int> offsets;
  tf::buffer<int> data;

  tf::generate_offset_blocks(
      tf::make_sequence_range(3), offsets, data,
      [](int value, tf::buffer<int> &block) { block.push_back(value); });
  REQUIRE(offsets.size() == 4);
  CHECK(offsets[0] == 0);
  CHECK(offsets[3] == 3);
  CHECK(data.size() == 3);

  tf::generate_offset_blocks(
      tf::make_sequence_range(0), offsets, data,
      [](int value, tf::buffer<int> &block) { block.push_back(value); });
  CHECK(offsets.size() == 0);
}

/// The checked tag is uniform across the vocabulary, and the iterator form of
/// `parallel_for` takes it like every other: below the cutoff the callable sees
/// the whole span once, and above it the same callable sees the same span in
/// pieces.
TEST_CASE("the iterator parallel_for takes the checked tag",
          "[core][algorithm][parallel]") {
  std::atomic<int> calls{0};
  std::atomic<std::size_t> visited{0};
  const auto span = [&](std::size_t begin, std::size_t end) {
    calls.fetch_add(1, std::memory_order_relaxed);
    visited.fetch_add(end - begin, std::memory_order_relaxed);
  };

  tf::parallel_for(std::size_t{0}, std::size_t{16}, span, tf::checked);
  CHECK(calls.load() == 1);
  CHECK(visited.load() == 16);

  calls.store(0);
  visited.store(0);
  tf::parallel_for(std::size_t{0}, std::size_t{1 << 16}, span, tf::checked);
  CHECK(calls.load() >= 1);
  CHECK(visited.load() == std::size_t{1} << 16);
}

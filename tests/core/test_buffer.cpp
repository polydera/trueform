/**
 * @file test_buffer.cpp
 * @brief Tests for tf::buffer capacity semantics.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core.hpp>

TEST_CASE("buffer reserve is a no-op when capacity suffices",
          "[core][buffer]") {
  tf::buffer<int> b;
  b.reserve(100);
  REQUIRE(b.capacity() >= 100);
  REQUIRE(b.size() == 0);
  const auto *data = b.data();
  const auto cap = b.capacity();
  for (int i = 0; i < 100; ++i)
    b.push_back(i);
  b.clear();
  b.reserve(50);
  REQUIRE(b.data() == data);
  REQUIRE(b.capacity() == cap);
  b.reserve(100);
  REQUIRE(b.data() == data);
  REQUIRE(b.capacity() == cap);
  b.reserve(cap + 1);
  REQUIRE(b.capacity() >= cap + 1);
}

TEST_CASE("buffer reserve preserves content while growing", "[core][buffer]") {
  tf::buffer<int> b;
  for (int i = 0; i < 10; ++i)
    b.push_back(i);
  b.reserve(b.capacity() + 100);
  REQUIRE(b.size() == 10);
  for (int i = 0; i < 10; ++i)
    REQUIRE(b[std::size_t(i)] == i);
}

TEST_CASE("buffer reallocate_and_initialize fills new elements within "
          "capacity",
          "[core][buffer]") {
  tf::buffer<int> b;
  b.reserve(64);
  b.push_back(1);
  b.push_back(2);
  b.reallocate_and_initialize(10, 7);
  REQUIRE(b.size() == 10);
  REQUIRE(b[0] == 1);
  REQUIRE(b[1] == 2);
  for (std::size_t i = 2; i < 10; ++i)
    REQUIRE(b[i] == 7);
  // growth branch fills too
  b.reallocate_and_initialize(b.capacity() + 8, 9);
  for (std::size_t i = 10; i < b.size(); ++i)
    REQUIRE(b[i] == 9);
}

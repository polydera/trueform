/**
 * @file test_blocked_buffer.cpp
 * @brief Tests for tf::blocked_buffer, static and dynamic block size.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core.hpp>

TEST_CASE("blocked_buffer static block size basics", "[core][blocked_buffer]") {
  tf::blocked_buffer<int, 3> b;
  b.emplace_back(0, 1, 2);
  b.emplace_back(3, 4, 5);
  REQUIRE(b.size() == 2);
  auto [x, y, z] = b[1];
  REQUIRE((x == 3 && y == 4 && z == 5));
}

TEST_CASE("blocked_buffer dynamic block size", "[core][blocked_buffer]") {
  tf::blocked_buffer<int, tf::dynamic_size> b(4);
  REQUIRE(b.block_size() == 4);
  REQUIRE(b.empty());

  b.allocate(3);
  REQUIRE(b.size() == 3);
  for (std::size_t k = 0; k < 3; ++k)
    for (std::size_t i = 0; i < 4; ++i)
      b[k][i] = int(k * 4 + i);

  SECTION("indexing and iteration see block views") {
    REQUIRE(b[2][0] == 8);
    REQUIRE(b.front()[3] == 3);
    REQUIRE(b.back()[3] == 11);
    std::size_t k = 0;
    for (auto blk : b) {
      REQUIRE(std::size_t(blk.size()) == 4);
      REQUIRE(blk[0] == int(k * 4));
      ++k;
    }
    REQUIRE(k == 3);
  }

  SECTION("push_back of a range appends one block") {
    int extra[4] = {100, 101, 102, 103};
    b.push_back(tf::make_range(extra, extra + 4));
    REQUIRE(b.size() == 4);
    REQUIRE(b[3][2] == 102);
  }

  SECTION("generic core helpers dispatch to the specialization") {
    tf::core::allocate(b, 5);
    REQUIRE(b.size() == 5);
    tf::core::reallocate(b, 2);
    REQUIRE(b.size() == 2);
  }

  SECTION("move keeps data and block size") {
    auto moved = std::move(b);
    REQUIRE(moved.block_size() == 4);
    REQUIRE(moved.size() == 3);
    REQUIRE(moved[1][1] == 5);
  }

  SECTION("adopting a flat buffer") {
    tf::buffer<int> flat;
    for (int i = 0; i < 6; ++i)
      flat.push_back(i);
    tf::blocked_buffer<int, tf::dynamic_size> adopted(std::move(flat), 2);
    REQUIRE(adopted.size() == 3);
    REQUIRE(adopted[2][1] == 5);
  }
}

TEST_CASE("blocked_buffer dynamic default construction is empty",
          "[core][blocked_buffer]") {
  tf::blocked_buffer<int, tf::dynamic_size> b;
  REQUIRE(b.block_size() == 0);
  REQUIRE(b.size() == 0);
  REQUIRE(b.empty());
}

/**
 * @file test_aabb_tree.cpp
 * @brief Tests for AABB tree construction
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/views/enumerate.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/spatial/search.hpp>
#include <vector>

namespace {

// Coordinates whose doubled centre is still representable: the largest
// centroid span an int64 entry can carry.
constexpr std::int64_t ceiling_coord = (std::int64_t(1) << 62) - 1;
// Coordinates whose doubled centre is not: the producer must not overflow.
constexpr std::int64_t over_coord = 5000000000000000000LL;

auto scattered(std::size_t n, std::int64_t range, std::uint64_t seed)
    -> tf::points_buffer<std::int64_t, 2> {
  tf::points_buffer<std::int64_t, 2> points;
  points.reserve(n);
  std::uint64_t s = seed;
  auto next = [&]() -> std::int64_t {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    // Magnitude on an exact integer grid over [0, range], sign from one bit:
    // the fixture must span both halves without overflowing while generating.
    auto magnitude =
        std::int64_t(range / (std::int64_t(1) << 31) * std::int64_t(s >> 33));
    return (s & 1) ? magnitude : -magnitude;
  };
  for (std::size_t i = 0; i < n; ++i) {
    auto x = next();
    auto y = next();
    points.emplace_back(x, y);
  }
  return points;
}

template <typename Tree>
auto ids_are_a_permutation(const Tree &tree, std::size_t n) -> bool {
  std::vector<int> seen(n, 0);
  for (auto id : tree.ids()) {
    if (id < 0 || std::size_t(id) >= n || seen[std::size_t(id)])
      return false;
    seen[std::size_t(id)] = 1;
  }
  return tree.ids().size() == n;
}

// Overlap is pure comparison: a query over coordinates this large must not
// take a difference or a product anywhere.
template <typename BV>
auto overlaps(const BV &bv, std::int64_t lo, std::int64_t hi) -> bool {
  for (std::size_t d = 0; d < 2; ++d)
    if (bv.max[d] < lo || hi < bv.min[d])
      return false;
  return true;
}

template <typename Tree>
auto searched_ids(const Tree &tree, std::int64_t lo, std::int64_t hi)
    -> std::vector<int> {
  std::vector<int> hits;
  tf::search(
      tree, [&](const auto &bv) { return overlaps(bv, lo, hi); },
      [&](auto id) { hits.push_back(int(id)); });
  std::sort(hits.begin(), hits.end());
  return hits;
}

template <typename Tree>
auto brute_force_ids(const Tree &tree, std::int64_t lo, std::int64_t hi)
    -> std::vector<int> {
  std::vector<int> hits;
  for (auto [i, bv] : tf::enumerate(tree.primitive_aabbs()))
    if (overlaps(bv, lo, hi))
      hits.push_back(int(i));
  return hits;
}

template <typename Tree>
auto queries_match_brute_force(const Tree &tree, std::int64_t range) -> bool {
  const std::int64_t probes[][2] = {{-range, range},    {-range, 0},
                                    {0, range},         {-range / 3, range / 7},
                                    {range / 2, range}, {range, range}};
  for (const auto &probe : probes)
    if (searched_ids(tree, probe[0], probe[1]) !=
        brute_force_ids(tree, probe[0], probe[1]))
      return false;
  return true;
}

} // namespace

TEST_CASE("integral centroid extent selects the mathematical largest axis",
          "[spatial][aabb_tree]") {
  constexpr std::int64_t far_x = 4000000000000000000LL;
  constexpr std::int64_t far_y = 1500000000000000000LL;

  tf::points_buffer<std::int64_t, 2> points;
  points.reserve(5);
  points.emplace_back(-far_x, -far_y);
  points.emplace_back(-far_x / 2, -far_y / 2);
  points.emplace_back(0, 0);
  points.emplace_back(far_x / 2, far_y / 2);
  points.emplace_back(far_x, far_y);

  tf::aabb_tree<int, std::int64_t, 2> tree(points.points(),
                                           tf::config_tree(4, 4));

  REQUIRE(tree.nodes()[0].axis == 0);
}

TEST_CASE("centroid span spanning the whole coordinate width selects its axis",
          "[spatial][aabb_tree]") {
  tf::points_buffer<std::int64_t, 2> points;
  points.reserve(5);
  points.emplace_back(-ceiling_coord, -1);
  points.emplace_back(-ceiling_coord / 2, 0);
  points.emplace_back(0, 1);
  points.emplace_back(ceiling_coord / 2, 0);
  points.emplace_back(ceiling_coord, -1);

  tf::aabb_tree<int, std::int64_t, 2> tree(points.points(),
                                           tf::config_tree(4, 4));

  REQUIRE(tree.nodes()[0].axis == 0);
}

TEST_CASE("doubled centroid past the coordinate type still selects the "
          "spread axis",
          "[spatial][aabb_tree]") {
  tf::points_buffer<std::int64_t, 2> points;
  points.reserve(5);
  points.emplace_back(-over_coord, 7);
  points.emplace_back(-over_coord / 2, 7);
  points.emplace_back(0, 7);
  points.emplace_back(over_coord / 2, 7);
  points.emplace_back(over_coord, 7);

  tf::aabb_tree<int, std::int64_t, 2> tree(points.points(),
                                           tf::config_tree(4, 4));

  REQUIRE(tree.nodes()[0].axis == 0);
}

TEST_CASE("doubled centroid past the coordinate type indexes every primitive",
          "[spatial][aabb_tree]") {
  auto points = scattered(4000, over_coord, 0x9E3779B97F4A7C15ull);
  tf::aabb_tree<int, std::int64_t, 2> tree(points.points(),
                                           tf::config_tree(4, 4));

  REQUIRE(ids_are_a_permutation(tree, points.size()));
  REQUIRE(queries_match_brute_force(tree, over_coord));
}

TEST_CASE("doubled centroid past the coordinate type indexes every primitive "
          "on the parallel split",
          "[spatial][aabb_tree]") {
  auto points = scattered(200000, over_coord, 0xD1B54A32D192ED03ull);
  tf::aabb_tree<int, std::int64_t, 2> tree(points.points(),
                                           tf::config_tree(4, 4));

  REQUIRE(ids_are_a_permutation(tree, points.size()));
  REQUIRE(queries_match_brute_force(tree, over_coord));
}

/**
 * @file test_hole_tiny_tier.cpp
 * @brief Tests for the enumerating tier and the overlap it admits on
 *
 * Tests for:
 * - tf::fill::fill_tiny_hole
 * - tf::fill::hole_triangles_overlap
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/fill/fill_holes.hpp>
#include <trueform/fill/holes/hole_triangles_overlap.hpp>
#include "fill_generators.hpp"
#include <array>
#include <cstddef>
#include <vector>

namespace {

using tiny_index = int;
using tiny_int = tf::exact::int32;
using tiny_real = float;

auto tiny_holds_chord(const tf::hole_fill_result<tiny_index, tiny_real> &result,
                      tiny_index a, tiny_index b) -> bool {
  for (const auto &triangle : result.triangles[0])
    for (int e = 0; e < 3; ++e) {
      const tiny_index x = triangle[std::size_t(e)];
      const tiny_index y = triangle[std::size_t((e + 1) % 3)];
      if ((x == a && y == b) || (x == b && y == a))
        return true;
    }
  return false;
}

} // namespace

TEST_CASE("fill: two coplanar triangles on one edge overlap when their "
          "apexes share a side",
          "[fill][tiny]") {
  tf::buffer<tf::point<tiny_int, 3>> points;
  points.push_back({0, 0, 0});
  points.push_back({4, 0, 0});
  points.push_back({1, 1, 0});
  points.push_back({0, 4, 0});
  const auto sites = tf::make_range(points);

  REQUIRE(tf::fill::hole_triangles_overlap(
      sites, std::array<tiny_index, 3>{1, 2, 3},
      std::array<tiny_index, 3>{1, 3, 0}));
  REQUIRE_FALSE(tf::fill::hole_triangles_overlap(
      sites, std::array<tiny_index, 3>{0, 1, 2},
      std::array<tiny_index, 3>{0, 2, 3}));
}

TEST_CASE("fill: the concave quad takes the chord that does not overlap",
          "[fill][tiny]") {
  const std::vector<tf::point<tiny_real, 3>> loop{
      {0.0f, 0.0f, 0.0f},
      {4.0f, 0.0f, 0.0f},
      {1.0f, 1.0f, 0.0f},
      {0.0f, 4.0f, 0.0f}};
  const auto mesh = hole_cone_mesh<tiny_index, tiny_real>(
      loop, tf::point<tiny_real, 3>{1.0f, 1.0f, -4.0f});
  const auto rims = tf::make_boundary_rims(mesh.polygons());
  REQUIRE(rims.size() == 1);

  const auto result = tf::fill_holes(mesh.polygons(), rims);
  REQUIRE(result.status[0] == tf::hole_fill_status::filled);
  REQUIRE(result.triangles[0].size() == 2);
  REQUIRE(tiny_holds_chord(result, 0, 2));
  REQUIRE_FALSE(tiny_holds_chord(result, 1, 3));
}

TEST_CASE("fill: a triangular rim is its own patch", "[fill][tiny]") {
  const std::vector<tf::point<tiny_real, 3>> loop{
      {0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {0.0f, 2.0f, 1.0f}};
  const auto mesh = hole_cone_mesh<tiny_index, tiny_real>(
      loop, tf::point<tiny_real, 3>{0.5f, 0.5f, -3.0f});
  const auto rims = tf::make_boundary_rims(mesh.polygons());
  const auto result = tf::fill_holes(mesh.polygons(), rims);
  REQUIRE(result.status[0] == tf::hole_fill_status::filled);
  REQUIRE(result.triangles[0].size() == 1);
  REQUIRE(result.refined[0] == tf::hole_refine_status::floor_met);
  REQUIRE(result.minted_points[0].size() == 0);
}

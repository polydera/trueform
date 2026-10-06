/**
 * @file test_hole_face_plans.cpp
 * @brief Tests for the canonical subdivision and the chords it forbids
 *
 * Tests for:
 * - tf::fill::subdivide_carrier_triangle
 * - tf::fill::hole_chord_is_forbidden
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/fill/holes/hole_chord_is_forbidden.hpp>
#include <trueform/fill/holes/subdivide_carrier_triangle.hpp>
#include <trueform/topology/face_membership.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace {

using plan_index = int;
using plan_int = tf::exact::int32;
using plan_real = float;
using plan_triangle = std::array<plan_index, 3>;

auto plan_split(plan_index face, plan_index v0, plan_index v1, plan_index point,
                int parameter) -> tf::hole_split<plan_index> {
  return {face, v0, v1, point, std::uint8_t(parameter)};
}

auto plan_doubled_area(const std::vector<std::array<int, 2>> &points,
                       const plan_triangle &triangle) -> int {
  const auto &a = points[std::size_t(triangle[0])];
  const auto &b = points[std::size_t(triangle[1])];
  const auto &c = points[std::size_t(triangle[2])];
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

/**
 * @brief A mesh of one triangle and one triangle beside it.
 */
auto plan_pair_mesh() -> tf::polygons_buffer<plan_index, plan_real, 3, 3> {
  tf::polygons_buffer<plan_index, plan_real, 3, 3> mesh;
  mesh.points_buffer().emplace_back(0.0f, 0.0f, 0.0f);
  mesh.points_buffer().emplace_back(4.0f, 0.0f, 0.0f);
  mesh.points_buffer().emplace_back(0.0f, 4.0f, 0.0f);
  mesh.points_buffer().emplace_back(4.0f, 4.0f, 0.0f);
  mesh.faces_buffer().emplace_back(plan_index(0), plan_index(1),
                                   plan_index(2));
  mesh.faces_buffer().emplace_back(plan_index(1), plan_index(3),
                                   plan_index(2));
  return mesh;
}

auto plan_rim_with_split(plan_index carrier, plan_index mint)
    -> tf::fill::hole_rim<plan_index, plan_int, plan_real> {
  tf::fill::hole_rim<plan_index, plan_int, plan_real> rim;
  rim.corners.push_back(0);
  rim.corners.push_back(mint);
  rim.edges.push_back({0, 0, 1, {0.0, 0.0, 1.0}, 0, 64, true});
  rim.edges.push_back({carrier, 1, 2, {0.0, 0.0, 1.0}, 32, 64, true});
  return rim;
}

} // namespace

TEST_CASE("fill: a carrier split on all three edges states its canonical "
          "subdivision",
          "[fill][plan]") {
  const std::vector<std::array<int, 2>> points{{0, 0}, {8, 0}, {0, 8}, {2, 0},
                                               {6, 0}, {0, 2}, {0, 6}, {6, 2},
                                               {2, 6}};
  tf::buffer<tf::hole_split<plan_index>> splits;
  splits.push_back(plan_split(0, 0, 1, 3, 16));
  splits.push_back(plan_split(0, 0, 1, 4, 48));
  splits.push_back(plan_split(0, 0, 2, 5, 16));
  splits.push_back(plan_split(0, 0, 2, 6, 48));
  splits.push_back(plan_split(0, 1, 2, 7, 16));
  splits.push_back(plan_split(0, 1, 2, 8, 48));

  tf::fill::hole_plan_scratch<plan_index> scratch;
  tf::buffer<plan_triangle> plan;
  tf::fill::subdivide_carrier_triangle(plan_triangle{0, 1, 2},
                                       tf::make_range(splits), scratch, plan);

  REQUIRE(plan.size() == 7);
  std::vector<int> areas;
  int total = 0;
  for (const auto &triangle : plan) {
    const int area = plan_doubled_area(points, triangle);
    REQUIRE(area > 0);
    areas.push_back(area);
    total += area;
  }
  std::sort(areas.begin(), areas.end());
  REQUIRE(areas == std::vector<int>{4, 4, 4, 4, 8, 8, 32});
  REQUIRE(total == 64);

  const std::vector<std::array<plan_index, 2>> subedges{
      {0, 3}, {3, 4}, {4, 1}, {1, 7}, {7, 8}, {8, 2}, {2, 6}, {6, 5}, {5, 0}};
  for (const auto &subedge : subedges) {
    int found = 0;
    for (const auto &triangle : plan)
      for (int e = 0; e < 3; ++e)
        if (triangle[std::size_t(e)] == subedge[0] &&
            triangle[std::size_t((e + 1) % 3)] == subedge[1])
          ++found;
    REQUIRE(found == 1);
  }
}

TEST_CASE("fill: a chord whose ends share a carrying triangle is forbidden",
          "[fill][plan]") {
  const auto mesh = plan_pair_mesh();
  tf::face_membership<plan_index> membership;
  membership.build(mesh.polygons());

  const auto shared = plan_rim_with_split(0, 9);
  REQUIRE(tf::fill::hole_chord_is_forbidden(mesh.polygons(), membership,
                                            shared, 0, 1));

  const auto apart = plan_rim_with_split(1, 9);
  REQUIRE_FALSE(tf::fill::hole_chord_is_forbidden(mesh.polygons(), membership,
                                                  apart, 0, 1));
}

TEST_CASE("fill: two splits of one carrying triangle may not be joined",
          "[fill][plan]") {
  const auto mesh = plan_pair_mesh();
  tf::face_membership<plan_index> membership;
  membership.build(mesh.polygons());

  tf::fill::hole_rim<plan_index, plan_int, plan_real> rim;
  rim.corners.push_back(9);
  rim.corners.push_back(10);
  rim.edges.push_back({0, 0, 1, {0.0, 0.0, 1.0}, 32, 64, true});
  rim.edges.push_back({0, 1, 2, {0.0, 0.0, 1.0}, 32, 64, true});
  REQUIRE(tf::fill::hole_chord_is_forbidden(mesh.polygons(), membership, rim,
                                            0, 1));

  rim.edges[1].face = 1;
  REQUIRE_FALSE(tf::fill::hole_chord_is_forbidden(mesh.polygons(), membership,
                                                  rim, 0, 1));
}

TEST_CASE("fill: an edge the mesh already winds is forbidden", "[fill][plan]") {
  const auto mesh = plan_pair_mesh();
  tf::face_membership<plan_index> membership;
  membership.build(mesh.polygons());

  tf::fill::hole_rim<plan_index, plan_int, plan_real> rim;
  rim.corners.push_back(1);
  rim.corners.push_back(2);
  rim.edges.push_back({0, 0, 1, {0.0, 0.0, 1.0}, 0, 64, true});
  rim.edges.push_back({0, 0, 2, {0.0, 0.0, 1.0}, 0, 64, true});
  REQUIRE(tf::fill::hole_chord_is_forbidden(mesh.polygons(), membership, rim,
                                            0, 1));

  rim.corners[0] = 0;
  rim.corners[1] = 3;
  REQUIRE_FALSE(tf::fill::hole_chord_is_forbidden(mesh.polygons(), membership,
                                                  rim, 0, 1));
}

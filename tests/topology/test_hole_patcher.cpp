/**
 * @file test_hole_patcher.cpp
 * @brief Tests for tf::hole_patcher with exact arithmetic
 *
 * Verifies hole patching: bridge edges, shared vertex splice,
 * multiple holes, degenerate cuts, and corner cases.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/offset_block_buffer.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/topology/hole_patcher.hpp>

using Index = int;

namespace {

auto make_pts(const std::vector<std::array<int32_t, 2>> &data)
    -> tf::points_buffer<int32_t, 2> {
  tf::points_buffer<int32_t, 2> pts;
  pts.allocate(data.size());
  for (std::size_t i = 0; i < data.size(); ++i) {
    pts[i][0] = data[i][0];
    pts[i][1] = data[i][1];
  }
  return pts;
}

auto make_holes(const std::vector<std::vector<int>> &hole_data)
    -> tf::offset_block_buffer<int, int> {
  tf::offset_block_buffer<int, int> obb;
  obb.offsets_buffer().push_back(0);
  for (auto &hole : hole_data) {
    for (auto v : hole)
      obb.data_buffer().push_back(v);
    obb.offsets_buffer().push_back(
        static_cast<int>(obb.data_buffer().size()));
  }
  return obb;
}

auto result_verts(const tf::hole_patcher<Index> &hp) -> std::vector<int> {
  std::vector<int> v;
  for (auto vid : hp.face())
    v.push_back(vid);
  return v;
}

} // namespace

TEST_CASE("Simple hole inside face", "[hole_patcher]") {
  auto pts = make_pts({{0, 0},
                       {200, 0},
                       {200, 200},
                       {0, 200},
                       {70, 70},
                       {130, 70},
                       {130, 130},
                       {70, 130}});
  std::vector<int> face = {0, 1, 2, 3};
  auto holes_obb = make_holes({{7, 6, 5, 4}});

  tf::hole_patcher<Index> hp;
  hp.build(tf::make_range(face), tf::make_faces(holes_obb), pts.points());

  CHECK(result_verts(hp) ==
        std::vector<int>{0, 1, 2, 3, 7, 6, 5, 4, 7, 3});
}

TEST_CASE("Hole sharing vertex with face", "[hole_patcher]") {
  auto pts = make_pts(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200}, {60, 40}, {40, 60}});
  std::vector<int> face = {0, 1, 2, 3};
  auto holes_obb = make_holes({{0, 5, 4}});

  tf::hole_patcher<Index> hp;
  hp.build(tf::make_range(face), tf::make_faces(holes_obb), pts.points());

  // Splice at shared vertex 0 — no extra bridge vertices
  CHECK(result_verts(hp) == std::vector<int>{0, 1, 2, 3, 0, 5, 4});
}

TEST_CASE("Multiple holes", "[hole_patcher]") {
  auto pts = make_pts({{0, 0},
                       {400, 0},
                       {400, 200},
                       {0, 200},
                       {50, 70},
                       {110, 70},
                       {110, 130},
                       {50, 130},
                       {200, 70},
                       {260, 70},
                       {260, 130},
                       {200, 130}});
  std::vector<int> face = {0, 1, 2, 3};
  auto holes_obb = make_holes({{7, 6, 5, 4}, {11, 10, 9, 8}});

  tf::hole_patcher<Index> hp;
  hp.build(tf::make_range(face), tf::make_faces(holes_obb), pts.points());

  CHECK(result_verts(hp) ==
        std::vector<int>{0, 1, 2, 3, 7, 6, 11, 10, 9, 8, 11, 6, 5, 4, 7, 3});
}

TEST_CASE("Degenerate cut hole (2 vertices)", "[hole_patcher]") {
  auto pts =
      make_pts({{0, 0}, {200, 0}, {200, 200}, {0, 200}, {100, 100}});
  std::vector<int> face = {0, 1, 2, 3};
  auto holes_obb = make_holes({{0, 4}});

  tf::hole_patcher<Index> hp;
  hp.build(tf::make_range(face), tf::make_faces(holes_obb), pts.points());

  // Splice at shared vertex 0
  CHECK(result_verts(hp) == std::vector<int>{0, 1, 2, 3, 0, 4});
}

TEST_CASE("Shared vertex NOT at hole leftmost", "[hole_patcher]") {
  auto pts = make_pts(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200}, {150, 50}, {170, 80}});
  std::vector<int> face = {0, 1, 2, 3};
  auto holes_obb = make_holes({{1, 5, 4}});

  tf::hole_patcher<Index> hp;
  hp.build(tf::make_range(face), tf::make_faces(holes_obb), pts.points());

  // find_shared_vertex pre-scan finds shared vertex 1
  CHECK(result_verts(hp) == std::vector<int>{0, 1, 5, 4, 1, 2, 3});
}

TEST_CASE("Hole touching face at corner vertex", "[hole_patcher]") {
  auto pts = make_pts(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200}, {160, 160}, {180, 140}});
  std::vector<int> face = {0, 1, 2, 3};
  auto holes_obb = make_holes({{2, 4, 5}});

  tf::hole_patcher<Index> hp;
  hp.build(tf::make_range(face), tf::make_faces(holes_obb), pts.points());

  // Splice at shared vertex 2
  CHECK(result_verts(hp) == std::vector<int>{0, 1, 2, 4, 5, 2, 3});
}

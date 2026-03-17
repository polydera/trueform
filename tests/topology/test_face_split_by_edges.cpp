/**
 * @file test_face_split_by_edges.cpp
 * @brief Tests for tf::face_split_by_edges with exact arithmetic
 *
 * Verifies face subdivision: crossings, nested crossings, loops,
 * cuts, non-crossings, and combinations. All geometry uses int32
 * exact arithmetic.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/blocked_buffer.hpp>
#include <trueform/core/edges.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/topology/face_split_by_edges.hpp>

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

auto make_undirected_edges(const std::vector<std::array<int, 2>> &edges)
    -> tf::blocked_buffer<int, 2> {
  tf::blocked_buffer<int, 2> buf;
  buf.allocate(edges.size());
  for (std::size_t i = 0; i < edges.size(); ++i) {
    buf[i][0] = edges[i][0];
    buf[i][1] = edges[i][1];
  }
  return buf;
}

auto face_verts(const tf::face_split_by_edges<Index> &fsbe, std::size_t idx)
    -> std::vector<int> {
  std::vector<int> v;
  auto it = fsbe.faces().begin();
  std::advance(it, idx);
  for (auto vid : *it)
    v.push_back(vid);
  return v;
}

auto hole_verts(const tf::face_split_by_edges<Index> &fsbe, std::size_t idx)
    -> std::vector<int> {
  std::vector<int> v;
  auto it = fsbe.holes().begin();
  std::advance(it, idx);
  for (auto vid : *it)
    v.push_back(vid);
  return v;
}

auto face_area(const tf::face_split_by_edges<Index> &fsbe, std::size_t idx)
    -> long long {
  auto it = fsbe.face_areas().begin();
  std::advance(it, idx);
  return static_cast<long long>(*it);
}

auto hole_area(const tf::face_split_by_edges<Index> &fsbe, std::size_t idx)
    -> long long {
  auto it = fsbe.hole_areas().begin();
  std::advance(it, idx);
  return static_cast<long long>(*it);
}

} // namespace

TEST_CASE("Simple crossing", "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0}, {200, 0}, {200, 200}, {0, 200}, {100, 100}});
  auto edges = make_undirected_edges({{0, 4}, {4, 2}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 2);
  CHECK(fsbe.holes().size() == 0);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 4, 2, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{0, 1, 2, 4});
  CHECK(face_area(fsbe, 0) == 40000);
  CHECK(face_area(fsbe, 1) == 40000);
}

TEST_CASE("Nested crossings (different start/end)",
          "[face_split_by_edges]") {
  auto pts = make_pts({{0, 50},   {50, 0},   {150, 0},
                       {200, 50}, {150, 100}, {50, 100},
                       {75, 50},  {125, 50}});
  auto edges = make_undirected_edges({{0, 6}, {6, 5}, {1, 7}, {7, 4}});
  std::vector<int> face = {0, 1, 2, 3, 4, 5};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 3);
  CHECK(fsbe.holes().size() == 0);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 6, 5});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{0, 1, 7, 4, 5, 6});
  CHECK(face_verts(fsbe, 2) == std::vector<int>{1, 2, 3, 4, 7});
  CHECK(face_area(fsbe, 0) == 3750);
  CHECK(face_area(fsbe, 1) == 13750);
  CHECK(face_area(fsbe, 2) == 12500);
}

TEST_CASE("Same end crossings", "[face_split_by_edges]") {
  auto pts = make_pts(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200}, {60, 100}, {140, 100}});
  auto edges = make_undirected_edges({{0, 4}, {4, 2}, {1, 5}, {5, 2}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 3);
  CHECK(fsbe.holes().size() == 0);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 4, 2, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{0, 1, 5, 2, 4});
  CHECK(face_verts(fsbe, 2) == std::vector<int>{1, 2, 5});
  CHECK(face_area(fsbe, 0) == 32000);
  CHECK(face_area(fsbe, 1) == 36000);
  CHECK(face_area(fsbe, 2) == 12000);
}

TEST_CASE("Same start crossings", "[face_split_by_edges]") {
  auto pts = make_pts(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200}, {60, 100}, {140, 100}});
  auto edges = make_undirected_edges({{0, 4}, {4, 3}, {0, 5}, {5, 2}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 3);
  CHECK(fsbe.holes().size() == 0);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 4, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{0, 5, 2, 3, 4});
  CHECK(face_verts(fsbe, 2) == std::vector<int>{0, 1, 2, 5});
  CHECK(face_area(fsbe, 0) == 12000);
  CHECK(face_area(fsbe, 1) == 36000);
  CHECK(face_area(fsbe, 2) == 32000);
}

TEST_CASE("Same start AND end (area ordering)", "[face_split_by_edges]") {
  auto pts = make_pts(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200}, {100, 40}, {100, 160}});
  auto edges = make_undirected_edges({{0, 4}, {4, 2}, {0, 5}, {5, 2}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 3);
  CHECK(fsbe.holes().size() == 0);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 5, 2, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{0, 4, 2, 5});
  CHECK(face_verts(fsbe, 2) == std::vector<int>{0, 1, 2, 4});
  CHECK(face_area(fsbe, 0) == 28000);
  CHECK(face_area(fsbe, 1) == 24000);
  CHECK(face_area(fsbe, 2) == 28000);
}

TEST_CASE("Interior loop (not touching base)", "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0},
                       {200, 0},
                       {200, 200},
                       {0, 200},
                       {70, 70},
                       {130, 70},
                       {130, 130},
                       {70, 130}});
  auto edges = make_undirected_edges({{4, 5}, {5, 6}, {6, 7}, {7, 4}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 2);
  CHECK(fsbe.holes().size() == 1);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 1, 2, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{5, 6, 7, 4});
  CHECK(hole_verts(fsbe, 0) == std::vector<int>{4, 7, 6, 5});
  CHECK(face_area(fsbe, 0) == 80000);
  CHECK(face_area(fsbe, 1) == 7200);
  CHECK(hole_area(fsbe, 0) == -7200);

  // hole 0 assigned to face 0
  auto hff = fsbe.holes_for_faces();
  auto it = hff.begin();
  CHECK((*it).size() == 1);
  CHECK((*it)[0] == 0);
}

TEST_CASE("Loop sharing vertex with base (start==end)",
          "[face_split_by_edges]") {
  auto pts = make_pts(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200}, {40, 40}, {80, 40}, {60, 80}});
  auto edges = make_undirected_edges({{0, 4}, {4, 5}, {5, 6}, {6, 0}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 2);
  CHECK(fsbe.holes().size() == 1);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 1, 2, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{4, 5, 6, 0});
  CHECK(hole_verts(fsbe, 0) == std::vector<int>{0, 6, 5, 4});
  CHECK(face_area(fsbe, 0) == 80000);
  CHECK(face_area(fsbe, 1) == 2400);
  CHECK(hole_area(fsbe, 0) == -2400);
}

TEST_CASE("Crossing + loop sharing start vertex",
          "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0},
                       {200, 0},
                       {200, 200},
                       {0, 200},
                       {30, 150},
                       {50, 10},
                       {80, 10},
                       {65, 30}});
  auto edges = make_undirected_edges(
      {{0, 4}, {4, 2}, {0, 5}, {5, 6}, {6, 7}, {7, 0}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 3);
  CHECK(fsbe.holes().size() == 1);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 4, 2, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{0, 1, 2, 4});
  CHECK(face_verts(fsbe, 2) == std::vector<int>{5, 6, 7, 0});
  CHECK(hole_verts(fsbe, 0) == std::vector<int>{0, 7, 6, 5});
  CHECK(face_area(fsbe, 0) == 16000);
  CHECK(face_area(fsbe, 1) == 64000);
  CHECK(face_area(fsbe, 2) == 1450);
  CHECK(hole_area(fsbe, 0) == -1450);

  // hole assigned to face 1 (the larger triangle containing the loop)
  auto hff = fsbe.holes_for_faces();
  auto f0 = hff.begin();
  auto f1 = f0;
  ++f1;
  CHECK((*f0).size() == 0);
  CHECK((*f1).size() == 1);
  CHECK((*f1)[0] == 0);
}

TEST_CASE("Simple cut", "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0}, {200, 0}, {200, 200}, {0, 200}, {100, 100}});
  auto edges = make_undirected_edges({{0, 4}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 1);
  CHECK(fsbe.holes().size() == 1);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 1, 2, 3});
  CHECK(hole_verts(fsbe, 0) == std::vector<int>{0, 4});
  CHECK(face_area(fsbe, 0) == 80000);
  CHECK(hole_area(fsbe, 0) == 0);
}

TEST_CASE("Non-crossings (T-junction)", "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0}, {200, 0}, {200, 200}, {0, 200}, {100, 100}});
  auto edges = make_undirected_edges({{0, 4}, {4, 2}, {4, 3}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 3);
  CHECK(fsbe.holes().size() == 0);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 4, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{4, 0, 1, 2});
  CHECK(face_verts(fsbe, 2) == std::vector<int>{4, 2, 3});
  CHECK(face_area(fsbe, 0) == 20000);
  CHECK(face_area(fsbe, 1) == 40000);
  CHECK(face_area(fsbe, 2) == 20000);
}

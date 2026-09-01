/**
 * @file test_planar_graph_regions.cpp
 * @brief Tests for tf::planar_graph_regions with exact orient2d-based sort
 *
 * Verifies region extraction from planar graphs: correct region count,
 * correct winding (CCW = positive area for interior, CW = negative for
 * exterior), and correct vertex sequences.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/blocked_buffer.hpp>
#include <trueform/core/edges.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/exact/int128.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/topology/planar_graph_regions.hpp>

using planar_regions_index_t = int;
using planar_regions_int_t = tf::exact::int32;

namespace {

// Helper: build points buffer from {x, y} pairs
auto make_planar_regions_pts(const std::vector<std::array<int32_t, 2>> &data)
    -> tf::points_buffer<int32_t, 2> {
  tf::points_buffer<int32_t, 2> pts;
  pts.allocate(data.size());
  for (std::size_t i = 0; i < data.size(); ++i) {
    pts[i][0] = data[i][0];
    pts[i][1] = data[i][1];
  }
  return pts;
}

// Helper: build directed edges (both directions per undirected edge)
auto make_planar_regions_directed_edges(
    const std::vector<std::array<int, 2>> &undirected)
    -> tf::blocked_buffer<int, 2> {
  tf::blocked_buffer<int, 2> buf;
  buf.allocate(undirected.size() * 2);
  for (std::size_t i = 0; i < undirected.size(); ++i) {
    buf[2 * i][0] = undirected[i][0];
    buf[2 * i][1] = undirected[i][1];
    buf[2 * i + 1][0] = undirected[i][1];
    buf[2 * i + 1][1] = undirected[i][0];
  }
  return buf;
}

// Count regions with positive/negative signed area (int128 exact)
auto count_regions(const tf::planar_graph_regions<planar_regions_index_t,
                                                  planar_regions_int_t> &pgr,
                   const tf::points_buffer<int32_t, 2> &pts)
    -> std::pair<int, int> {
  int pos = 0, neg = 0;
  for (auto region : pgr) {
    tf::exact::int128 area2 = 0;
    for (std::size_t i = 0; i < region.size(); ++i) {
      auto j = (i + 1) % region.size();
      auto &&p0 = pts[region[i]];
      auto &&p1 = pts[region[j]];
      area2 += tf::exact::int128(int64_t(p1[1]) + int64_t(p0[1])) *
               tf::exact::int128(int64_t(p0[0]) - int64_t(p1[0]));
    }
    if (area2 > 0)
      ++pos;
    else if (area2 < 0)
      ++neg;
  }
  return {pos, neg};
}

// Collect vertex IDs of a specific region
auto region_verts(const tf::planar_graph_regions<planar_regions_index_t,
                                                 planar_regions_int_t> &pgr,
                  std::size_t idx) -> std::vector<int> {
  std::vector<int> v;
  auto it = pgr.begin();
  std::advance(it, idx);
  for (auto vid : *it)
    v.push_back(vid);
  return v;
}

} // namespace

TEST_CASE("Square with diagonal", "[planar_graph_regions]") {
  auto pts = make_planar_regions_pts({{0, 0}, {100, 0}, {100, 100}, {0, 100}});
  auto edges = make_planar_regions_directed_edges(
      {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 2}});

  tf::planar_graph_regions<planar_regions_index_t, planar_regions_int_t> pgr;
  pgr.build(tf::make_edges(edges), pts.points());

  auto [pos, neg] = count_regions(pgr, pts);
  CHECK(pgr.size() == 3);
  CHECK(pos == 2);
  CHECK(neg == 1);

  // Two triangles: (0,1,2) and (2,3,0), one exterior
  CHECK(region_verts(pgr, 0) == std::vector<int>{0, 1, 2});
  CHECK(region_verts(pgr, 2) == std::vector<int>{2, 3, 0});
}

TEST_CASE("Square with cross (X)", "[planar_graph_regions]") {
  auto pts = make_planar_regions_pts(
      {{0, 0}, {100, 0}, {100, 100}, {0, 100}, {50, 50}});
  auto edges = make_planar_regions_directed_edges(
      {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4}, {2, 4}, {3, 4}});

  tf::planar_graph_regions<planar_regions_index_t, planar_regions_int_t> pgr;
  pgr.build(tf::make_edges(edges), pts.points());

  auto [pos, neg] = count_regions(pgr, pts);
  CHECK(pgr.size() == 5);
  CHECK(pos == 4);
  CHECK(neg == 1);
}

TEST_CASE("T-junction", "[planar_graph_regions]") {
  auto pts = make_planar_regions_pts(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200}, {0, 100}, {100, 100}});
  auto edges = make_planar_regions_directed_edges(
      {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {4, 5}, {5, 1}});

  tf::planar_graph_regions<planar_regions_index_t, planar_regions_int_t> pgr;
  pgr.build(tf::make_edges(edges), pts.points());

  auto [pos, neg] = count_regions(pgr, pts);
  CHECK(pgr.size() == 3);
  CHECK(pos == 2);
  CHECK(neg == 1);

  // Bottom region: 0,1,5,4
  CHECK(region_verts(pgr, 0) == std::vector<int>{0, 1, 5, 4});
}

TEST_CASE("Nearly-collinear edges", "[planar_graph_regions]") {
  auto pts =
      make_planar_regions_pts({{0, 0}, {1000000, 0}, {500000, 1}, {500000, 2}});
  auto edges = make_planar_regions_directed_edges(
      {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 2}});

  tf::planar_graph_regions<planar_regions_index_t, planar_regions_int_t> pgr;
  pgr.build(tf::make_edges(edges), pts.points());

  auto [pos, neg] = count_regions(pgr, pts);
  CHECK(pgr.size() == 3);
  CHECK(pos == 2);
  CHECK(neg == 1);
}

TEST_CASE("Nested squares disconnected", "[planar_graph_regions]") {
  auto pts = make_planar_regions_pts({{0, 0},
                                      {200, 0},
                                      {200, 200},
                                      {0, 200},
                                      {50, 50},
                                      {150, 50},
                                      {150, 150},
                                      {50, 150}});
  auto edges = make_planar_regions_directed_edges(
      {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}});

  tf::planar_graph_regions<planar_regions_index_t, planar_regions_int_t> pgr;
  pgr.build(tf::make_edges(edges), pts.points());

  auto [pos, neg] = count_regions(pgr, pts);
  CHECK(pgr.size() == 4);
  CHECK(pos == 2);
  CHECK(neg == 2);
}

TEST_CASE("Nested squares connected", "[planar_graph_regions]") {
  auto pts = make_planar_regions_pts({{0, 0},
                                      {200, 0},
                                      {200, 200},
                                      {0, 200},
                                      {50, 50},
                                      {150, 50},
                                      {150, 150},
                                      {50, 150}});
  auto edges = make_planar_regions_directed_edges({{0, 1},
                                                   {1, 2},
                                                   {2, 3},
                                                   {3, 0},
                                                   {4, 5},
                                                   {5, 6},
                                                   {6, 7},
                                                   {7, 4},
                                                   {0, 4},
                                                   {1, 5},
                                                   {2, 6},
                                                   {3, 7}});

  tf::planar_graph_regions<planar_regions_index_t, planar_regions_int_t> pgr;
  pgr.build(tf::make_edges(edges), pts.points());

  auto [pos, neg] = count_regions(pgr, pts);
  CHECK(pgr.size() == 6);
  CHECK(pos == 5);
  CHECK(neg == 1);

  // Bottom quad: 0,1,5,4
  CHECK(region_verts(pgr, 0) == std::vector<int>{0, 1, 5, 4});
  // Inner square: 4,5,6,7
  CHECK(region_verts(pgr, 5) == std::vector<int>{4, 5, 6, 7});
}

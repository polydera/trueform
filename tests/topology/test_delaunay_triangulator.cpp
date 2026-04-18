/**
 * @file test_delaunay_triangulator.cpp
 * @brief Tests for tf::delaunay_triangulator
 *
 * Same invariants as test_ear_cutter.cpp (distinct, all_present,
 * edges, area) verified across all cyclic rotations.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/exact/int128.hpp>
#include <trueform/exact/signed_area.hpp>
#include <trueform/topology/delaunay_triangulator.hpp>

#include <map>
#include <vector>

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

auto verify_edges(int n, const tf::buffer<int> &tris) -> bool {
  using edge = std::pair<int, int>;
  std::map<edge, int> loop_counts, tri_counts;

  for (int i = 0; i < n; ++i)
    loop_counts[{i, (i + 1) % n}]++;

  for (std::size_t i = 0; i < tris.size(); i += 3) {
    int a = tris[i], b = tris[i + 1], c = tris[i + 2];
    tri_counts[{a, b}]++;
    tri_counts[{b, c}]++;
    tri_counts[{c, a}]++;
  }

  for (auto &[e, count] : tri_counts) {
    auto it = loop_counts.find(e);
    if (it != loop_counts.end()) {
      if (count != it->second)
        return false;
    } else {
      auto rev = tri_counts.find({e.second, e.first});
      if (rev == tri_counts.end() || count + rev->second != 2)
        return false;
    }
  }
  return true;
}

auto verify_area(int n, const tf::buffer<int> &tris,
                 const tf::points_buffer<int32_t, 2> &pts) -> bool {
  std::vector<int> ids(n);
  for (int i = 0; i < n; ++i)
    ids[i] = i;
  auto poly_area = tf::exact::signed_area_2x(
      tf::make_polygon(ids, pts.points()));
  tf::exact::int128 tri_area = 0;
  for (std::size_t i = 0; i < tris.size(); i += 3) {
    auto a = tris[i], b = tris[i + 1], c = tris[i + 2];
    auto &&pa = pts[a];
    auto &&pb = pts[b];
    auto &&pc = pts[c];
    tri_area += tf::exact::int128(int64_t(pb[0]) - int64_t(pa[0])) *
                    tf::exact::int128(int64_t(pc[1]) - int64_t(pa[1])) -
                tf::exact::int128(int64_t(pb[1]) - int64_t(pa[1])) *
                    tf::exact::int128(int64_t(pc[0]) - int64_t(pa[0]));
  }
  return poly_area == tri_area;
}

auto verify_all_present(int n, const tf::buffer<int> &tris) -> bool {
  for (int id = 0; id < n; ++id) {
    bool found = false;
    for (std::size_t i = 0; i < tris.size(); ++i)
      if (tris[i] == id) {
        found = true;
        break;
      }
    if (!found)
      return false;
  }
  return true;
}

auto verify_distinct(const tf::buffer<int> &tris) -> bool {
  for (std::size_t i = 0; i < tris.size(); i += 3) {
    auto a = tris[i], b = tris[i + 1], c = tris[i + 2];
    if (a == b || b == c || a == c)
      return false;
  }
  return true;
}

auto check_all_rotations(
    const std::vector<std::array<int32_t, 2>> &data) {
  int n = data.size();
  for (int rot = 0; rot < n; ++rot) {
    std::vector<std::array<int32_t, 2>> rotated(n);
    for (int i = 0; i < n; ++i)
      rotated[i] = data[(i + rot) % n];

    auto pts = make_pts(rotated);

    tf::delaunay_triangulator<int> dt;
    dt.build(pts.points());
    auto &tris = dt.indices_buffer();

    CAPTURE(rot);
    CHECK(verify_distinct(tris));
    CHECK(verify_all_present(n, tris));
    CHECK(verify_edges(n, tris));
    CHECK(verify_area(n, tris, pts));
  }
}

} // namespace

TEST_CASE("Delaunay 5 consecutive collinear points", "[delaunay]") {
  check_all_rotations(
      {{0, 0}, {25, 0}, {50, 0}, {75, 0}, {100, 0}, {50, 80}});
}

TEST_CASE("Delaunay coincident vertex pairs", "[delaunay]") {
  check_all_rotations({{469890606, 38424444},
                       {-471815356, 14819068},
                       {-473023691, -37},
                       {-473023691, -37},
                       {473023700, -37},
                       {473023700, -37},
                       {470228476, 34280776}});
}

TEST_CASE("Delaunay convex pentagon", "[delaunay]") {
  check_all_rotations(
      {{0, 0}, {100, 0}, {120, 60}, {50, 100}, {-20, 60}});
}

TEST_CASE("Delaunay L-shape", "[delaunay]") {
  check_all_rotations(
      {{0, 0}, {60, 0}, {60, 40}, {20, 40}, {20, 80}, {0, 80}});
}

TEST_CASE("Delaunay triangle", "[delaunay]") {
  check_all_rotations({{0, 0}, {100, 0}, {50, 80}});
}

TEST_CASE("Delaunay quad", "[delaunay]") {
  check_all_rotations({{0, 0}, {100, 0}, {100, 100}, {0, 100}});
}

TEST_CASE("Delaunay star shape", "[delaunay]") {
  check_all_rotations({{50, 100}, {30, 60}, {0, 50},
                       {30, 40},  {50, 0},  {70, 40},
                       {100, 50}, {70, 60}});
}

TEST_CASE("Delaunay near-collinear sliver", "[delaunay]") {
  check_all_rotations({{0, 0}, {100, 1}, {200, 2}, {100, 100}});
}

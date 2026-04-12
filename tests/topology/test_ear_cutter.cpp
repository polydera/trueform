/**
 * @file test_ear_cutter.cpp
 * @brief Tests for tf::ear_cutter with exact arithmetic
 *
 * Verifies triangulation invariants across all cyclic rotations:
 * distinct vertex identities, all vertices present, boundary/interior
 * edge counts, and signed area conservation. Tests cover bridges,
 * collinear runs, shards, and float-to-int32 conversion.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/exact/int128.hpp>
#include <trueform/exact/signed_area.hpp>
#include <trueform/topology/ear_cutter.hpp>

#include <map>
#include <type_traits>
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

auto make_float_pts(const std::vector<std::array<int32_t, 2>> &data)
    -> tf::points_buffer<float, 2> {
  tf::points_buffer<float, 2> pts;
  pts.allocate(data.size());
  for (std::size_t i = 0; i < data.size(); ++i) {
    pts[i][0] = float(data[i][0]);
    pts[i][1] = float(data[i][1]);
  }
  return pts;
}

// Verify edge invariants: boundary edges appear once, interior edges
// appear twice (once per direction).
auto verify_edges(const std::vector<int> &ids,
                  const tf::buffer<int> &tris) -> bool {
  using edge = std::pair<int, int>;
  std::map<edge, int> loop_counts, tri_counts;

  int n = ids.size();
  for (int i = 0; i < n; ++i)
    loop_counts[{ids[i], ids[(i + 1) % n]}]++;

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

// Verify signed area is conserved (int32 only).
auto verify_area(const std::vector<int> &ids,
                 const tf::buffer<int> &tris,
                 const tf::points_buffer<int32_t, 2> &pts) -> bool {
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

// Verify all input vertex IDs appear in the output.
auto verify_all_present(const std::vector<int> &ids,
                        const tf::buffer<int> &tris) -> bool {
  for (int id : ids) {
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

// Verify no triangle has repeated vertex identities.
auto verify_distinct(const tf::buffer<int> &tris) -> bool {
  for (std::size_t i = 0; i < tris.size(); i += 3) {
    auto a = tris[i], b = tris[i + 1], c = tris[i + 2];
    if (a == b || b == c || a == c)
      return false;
  }
  return true;
}

// Run earcut on all cyclic rotations and check all invariants.
template <typename Points>
auto check_all_rotations(const std::vector<int> &base_ids,
                         const Points &pts) {
  int n = base_ids.size();
  for (int rot = 0; rot < n; ++rot) {
    std::vector<int> ids(n);
    for (int i = 0; i < n; ++i)
      ids[i] = base_ids[(i + rot) % n];

    tf::ear_cutter<int> ec;
    ec.build(ids, pts.points());
    auto &tris = ec.indices_buffer();

    CAPTURE(rot);
    CHECK(verify_distinct(tris));
    CHECK(verify_all_present(ids, tris));
    CHECK(verify_edges(ids, tris));
  }
}

// Same but also checks exact area (int32 only).
auto check_all_rotations_exact(const std::vector<int> &base_ids,
                               const tf::points_buffer<int32_t, 2> &pts) {
  int n = base_ids.size();
  for (int rot = 0; rot < n; ++rot) {
    std::vector<int> ids(n);
    for (int i = 0; i < n; ++i)
      ids[i] = base_ids[(i + rot) % n];

    tf::ear_cutter<int> ec;
    ec.build(ids, pts.points());
    auto &tris = ec.indices_buffer();

    CAPTURE(rot);
    CHECK(verify_distinct(tris));
    CHECK(verify_all_present(ids, tris));
    CHECK(verify_edges(ids, tris));
    CHECK(verify_area(ids, tris, pts));
  }
}

} // namespace

TEST_CASE("Bridge shard all rotations", "[ear_cutter]") {
  auto pts = make_pts(
      {{-20, 0}, {20, 0}, {20, 10}, {10, 20}, {30, 20}, {60, 0}, {20, 40}});
  std::vector<int> ids = {0, 1, 2, 3, 4, 2, 1, 5, 6};
  check_all_rotations_exact(ids, pts);
}

TEST_CASE("Bridge shard all rotations (float)", "[ear_cutter]") {
  auto pts = make_float_pts(
      {{-20, 0}, {20, 0}, {20, 10}, {10, 20}, {30, 20}, {60, 0}, {20, 40}});
  std::vector<int> ids = {0, 1, 2, 3, 4, 2, 1, 5, 6};
  check_all_rotations(ids, pts);
}

TEST_CASE("Bridge with collinear", "[ear_cutter]") {
  auto pts = make_pts({{-20, 0},
                       {0, 0},
                       {20, 0},
                       {20, 10},
                       {10, 20},
                       {30, 20},
                       {40, 0},
                       {60, 0},
                       {44, 16},
                       {28, 32},
                       {20, 40},
                       {12, 32},
                       {-4, 16}});
  std::vector<int> ids = {0, 1, 2, 3, 4, 5, 3, 2, 6, 7, 8, 9, 10, 11, 12};
  check_all_rotations_exact(ids, pts);
}

TEST_CASE("Bridge with collinear (float)", "[ear_cutter]") {
  auto pts = make_float_pts({{-20, 0},
                             {0, 0},
                             {20, 0},
                             {20, 10},
                             {10, 20},
                             {30, 20},
                             {40, 0},
                             {60, 0},
                             {44, 16},
                             {28, 32},
                             {20, 40},
                             {12, 32},
                             {-4, 16}});
  std::vector<int> ids = {0, 1, 2, 3, 4, 5, 3, 2, 6, 7, 8, 9, 10, 11, 12};
  check_all_rotations(ids, pts);
}

TEST_CASE("Triangle with 2-edge shard", "[ear_cutter]") {
  auto pts =
      make_pts({{0, 0}, {100, 0}, {50, 80}, {50, 50}, {50, 30}});
  std::vector<int> ids = {0, 1, 2, 3, 4, 3, 2};
  check_all_rotations_exact(ids, pts);
}

TEST_CASE("Triangle with 2-edge shard (float)", "[ear_cutter]") {
  auto pts =
      make_float_pts({{0, 0}, {100, 0}, {50, 80}, {50, 50}, {50, 30}});
  std::vector<int> ids = {0, 1, 2, 3, 4, 3, 2};
  check_all_rotations(ids, pts);
}

TEST_CASE("Two shards", "[ear_cutter]") {
  auto pts = make_pts(
      {{-10, 0}, {20, 40}, {80, 40}, {30, 30}, {70, 30}, {110, 0}, {50, 90}});
  std::vector<int> ids = {0, 1, 3, 1, 6, 2, 4, 2, 5};
  check_all_rotations_exact(ids, pts);
}

TEST_CASE("Two shards (float)", "[ear_cutter]") {
  auto pts = make_float_pts(
      {{-10, 0}, {20, 40}, {80, 40}, {30, 30}, {70, 30}, {110, 0}, {50, 90}});
  std::vector<int> ids = {0, 1, 3, 1, 6, 2, 4, 2, 5};
  check_all_rotations(ids, pts);
}

TEST_CASE("5 consecutive collinear points", "[ear_cutter]") {
  auto pts =
      make_pts({{0, 0}, {25, 0}, {50, 0}, {75, 0}, {100, 0}, {50, 80}});
  std::vector<int> ids = {0, 1, 2, 3, 4, 5};
  check_all_rotations_exact(ids, pts);
}

TEST_CASE("5 consecutive collinear points (float)", "[ear_cutter]") {
  auto pts =
      make_float_pts({{0, 0}, {25, 0}, {50, 0}, {75, 0}, {100, 0}, {50, 80}});
  std::vector<int> ids = {0, 1, 2, 3, 4, 5};
  check_all_rotations(ids, pts);
}

TEST_CASE("Coincident vertex pairs from arrangement", "[ear_cutter]") {
  // 7-vertex polygon from cylinder/sphere intersection (fc loop 249).
  // Two pairs of coincident 2D points: pt2==pt3, pt4==pt5.
  auto pts = make_pts({{469890606, 38424444},
                        {-471815356, 14819068},
                        {-473023691, -37},
                        {-473023691, -37},
                        {473023700, -37},
                        {473023700, -37},
                        {470228476, 34280776}});
  std::vector<int> ids = {0, 1, 2, 3, 4, 5, 6};
  check_all_rotations_exact(ids, pts);
}

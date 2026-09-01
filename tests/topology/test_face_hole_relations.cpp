/**
 * @file test_face_hole_relations.cpp
 * @brief Tests for tf::face_hole_relations with exact arithmetic
 *
 * Verifies hole-to-face assignment using exact orient2d point-in-polygon
 * and exact signed_area for area comparison.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/blocked_buffer.hpp>
#include <trueform/core/edges.hpp>
#include <trueform/core/faces.hpp>
#include <trueform/core/offset_block_buffer.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/exact/int128.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/topology/face_hole_relations.hpp>
#include <trueform/topology/planar_graph_regions.hpp>

using hole_relations_index_t = int;
using hole_relations_int_t = tf::exact::int32;

namespace {

auto make_hole_relations_pts(const std::vector<std::array<int32_t, 2>> &data)
    -> tf::points_buffer<int32_t, 2> {
  tf::points_buffer<int32_t, 2> pts;
  pts.allocate(data.size());
  for (std::size_t i = 0; i < data.size(); ++i) {
    pts[i][0] = data[i][0];
    pts[i][1] = data[i][1];
  }
  return pts;
}

auto make_hole_relations_directed_edges(
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

struct fhr_result {
  std::size_t n_faces;
  std::size_t n_holes;
  std::vector<std::vector<int>> holes_per_face;
};

auto run_fhr(const std::vector<std::array<int32_t, 2>> &pt_data,
             const std::vector<std::array<int, 2>> &edge_data) -> fhr_result {
  auto pts = make_hole_relations_pts(pt_data);
  auto edges = make_hole_relations_directed_edges(edge_data);

  tf::planar_graph_regions<hole_relations_index_t, hole_relations_int_t> pgr;
  pgr.build(tf::make_edges(edges), pts.points());

  tf::offset_block_buffer<hole_relations_index_t, hole_relations_index_t>
      faces_obb, holes_obb;
  faces_obb.offsets_buffer().push_back(0);
  holes_obb.offsets_buffer().push_back(0);

  for (auto region : pgr) {
    tf::exact::int128 area2 = 0;
    for (std::size_t i = 0; i < region.size(); ++i) {
      auto j = (i + 1) % region.size();
      auto &&p0 = pts[region[i]];
      auto &&p1 = pts[region[j]];
      area2 += tf::exact::int128(int64_t(p1[1]) + int64_t(p0[1])) *
               tf::exact::int128(int64_t(p0[0]) - int64_t(p1[0]));
    }
    auto &target = (area2 > 0) ? faces_obb : holes_obb;
    for (auto vid : region)
      target.data_buffer().push_back(vid);
    target.offsets_buffer().push_back(
        static_cast<hole_relations_index_t>(target.data_buffer().size()));
  }

  tf::face_hole_relations<hole_relations_index_t, hole_relations_int_t> fhr;
  fhr.build(tf::make_faces(faces_obb), tf::make_faces(holes_obb),
            pts.points());

  fhr_result result;
  result.n_faces = faces_obb.size();
  result.n_holes = holes_obb.size();
  result.holes_per_face.resize(result.n_faces);
  for (std::size_t fi = 0; fi < result.n_faces; ++fi)
    if (fi < fhr.size())
      for (auto hole_id : fhr[fi])
        result.holes_per_face[fi].push_back(hole_id);
  return result;
}

} // namespace

TEST_CASE("Connected nested squares (no hole assignments)",
          "[face_hole_relations]") {
  auto r = run_fhr(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200},
       {50, 50}, {150, 50}, {150, 150}, {50, 150}},
      {{0, 1}, {1, 2}, {2, 3}, {3, 0},
       {4, 5}, {5, 6}, {6, 7}, {7, 4},
       {0, 4}, {1, 5}, {2, 6}, {3, 7}});

  CHECK(r.n_faces == 5);
  CHECK(r.n_holes == 1);
  for (auto &h : r.holes_per_face)
    CHECK(h.empty());
}

TEST_CASE("Simple hole (inner square inside outer)",
          "[face_hole_relations]") {
  auto r = run_fhr(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200},
       {50, 50}, {150, 50}, {150, 150}, {50, 150}},
      {{0, 1}, {1, 2}, {2, 3}, {3, 0},
       {4, 5}, {5, 6}, {6, 7}, {7, 4}});

  CHECK(r.n_faces == 2);
  CHECK(r.n_holes == 2);
  CHECK(r.holes_per_face[0].size() == 1);
  CHECK(r.holes_per_face[0][0] == 1);
  CHECK(r.holes_per_face[1].empty());
}

TEST_CASE("Three nested squares", "[face_hole_relations]") {
  auto r = run_fhr(
      {{0, 0}, {300, 0}, {300, 300}, {0, 300},
       {50, 50}, {250, 50}, {250, 250}, {50, 250},
       {100, 100}, {200, 100}, {200, 200}, {100, 200}},
      {{0, 1}, {1, 2}, {2, 3}, {3, 0},
       {4, 5}, {5, 6}, {6, 7}, {7, 4},
       {8, 9}, {9, 10}, {10, 11}, {11, 8}});

  CHECK(r.n_faces == 3);
  CHECK(r.n_holes == 3);
  CHECK(r.holes_per_face[0].size() == 1);
  CHECK(r.holes_per_face[0][0] == 1);
  CHECK(r.holes_per_face[1].size() == 1);
  CHECK(r.holes_per_face[1][0] == 2);
  CHECK(r.holes_per_face[2].empty());
}

TEST_CASE("Hole in split face", "[face_hole_relations]") {
  auto r = run_fhr(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200},
       {0, 100}, {100, 100},
       {30, 130}, {80, 130}, {80, 170}, {30, 170}},
      {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0},
       {4, 5}, {5, 1},
       {6, 7}, {7, 8}, {8, 9}, {9, 6}});

  CHECK(r.n_faces == 3);
  CHECK(r.n_holes == 2);
  CHECK(r.holes_per_face[1].size() == 1);
  CHECK(r.holes_per_face[1][0] == 1);
  CHECK(r.holes_per_face[0].empty());
  CHECK(r.holes_per_face[2].empty());
}

TEST_CASE("Multiple holes in one face", "[face_hole_relations]") {
  auto r = run_fhr(
      {{0, 0}, {400, 0}, {400, 400}, {0, 400},
       {50, 50}, {350, 50}, {350, 350}, {50, 350},
       {80, 80}, {180, 80}, {180, 180}, {80, 180},
       {220, 80}, {320, 80}, {320, 180}, {220, 180}},
      {{0, 1}, {1, 2}, {2, 3}, {3, 0},
       {4, 5}, {5, 6}, {6, 7}, {7, 4},
       {8, 9}, {9, 10}, {10, 11}, {11, 8},
       {12, 13}, {13, 14}, {14, 15}, {15, 12}});

  CHECK(r.n_faces == 4);
  CHECK(r.n_holes == 4);
  CHECK(r.holes_per_face[0].size() == 1);
  CHECK(r.holes_per_face[0][0] == 1);
  CHECK(r.holes_per_face[1].size() == 2);
  CHECK(r.holes_per_face[2].empty());
  CHECK(r.holes_per_face[3].empty());
}

/**
 * @file test_face_cutter.cpp
 * @brief Tests for tf::face_cutter
 *
 * Verifies face cutting: 0-edge passthrough, 1-edge fast split,
 * interior cuts, parallel edges, and interior loops (holes).
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/cut/face_cutter.hpp>
#include <trueform/exact/projection_axes.hpp>
#include <trueform/intersect/graph/edge.hpp>

using Index = int;
using vertex_t = tf::intersect::graph::vertex<Index>;
using source = tf::intersect::graph::vertex_source;
using edge_t = tf::intersect::graph::edge<Index>;

namespace {

auto V(Index id) -> vertex_t { return {source::original, id, 0}; }
auto C(Index id) -> vertex_t { return {source::created, id, 0}; }

auto vless(const vertex_t &a, const vertex_t &b) -> bool {
  if (a.source != b.source) return a.source < b.source;
  return a.id < b.id;
}

auto canonicalize(std::vector<vertex_t> loop) -> std::vector<vertex_t> {
  if (loop.empty()) return loop;
  auto it = std::min_element(loop.begin(), loop.end(), vless);
  std::rotate(loop.begin(), it, loop.end());
  return loop;
}

auto canonical_faces(const std::vector<std::vector<vertex_t>> &faces)
    -> std::vector<std::vector<vertex_t>> {
  std::vector<std::vector<vertex_t>> result;
  for (auto &f : faces)
    result.push_back(canonicalize(f));
  std::sort(result.begin(), result.end(),
            [](const auto &a, const auto &b) {
              return std::lexicographical_compare(
                  a.begin(), a.end(), b.begin(), b.end(), vless);
            });
  return result;
}

struct test_geometry {
  std::vector<std::array<int32_t, 3>> pts;
  std::vector<int> face;

  auto get_point_3d(const vertex_t &v) const -> tf::point<int32_t, 3> {
    return {pts[v.id][0], pts[v.id][1], pts[v.id][2]};
  }

  auto make_get_point() const {
    auto p0 = get_point_3d(V(face[0]));
    auto p1 = get_point_3d(V(face[1]));
    auto p2 = get_point_3d(V(face[2]));
    auto axes = tf::exact::projection_axes(p0, p1, p2);
    return [this, axes](const vertex_t &v) -> tf::point<int32_t, 2> {
      auto [ax0, ax1] = axes;
      auto pt = get_point_3d(v);
      return {pt[ax0], pt[ax1]};
    };
  }
};

auto collect_faces(const tf::buffer<Index> &offsets,
                   const tf::buffer<vertex_t> &vertices)
    -> std::vector<std::vector<vertex_t>> {
  std::vector<std::vector<vertex_t>> result;
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    auto start = offsets[i];
    auto end = (i + 1 < offsets.size()) ? offsets[i + 1]
                                        : static_cast<Index>(vertices.size());
    std::vector<vertex_t> face;
    for (Index j = start; j < end; ++j)
      face.push_back(vertices[j]);
    result.push_back(face);
  }
  return result;
}

auto run(const test_geometry &geo, const std::vector<vertex_t> &loop,
         const std::vector<edge_t> &edges)
    -> std::vector<std::vector<vertex_t>> {
  tf::face_cutter<Index> fc;
  tf::buffer<Index> offsets;
  tf::buffer<vertex_t> vertices;
  fc.build(tf::make_range(loop), tf::make_range(edges),
           geo.make_get_point(), offsets, vertices);
  return canonical_faces(collect_faces(offsets, vertices));
}

} // namespace

TEST_CASE("No edges - emit base loop", "[face_cutter]") {
  test_geometry geo{{{0, 0, 0}, {200, 0, 0}, {200, 200, 0}, {0, 200, 0}},
                    {0, 1, 2, 3}};
  auto faces = run(geo, {V(0), V(1), V(2), V(3)}, {});
  auto expected = canonical_faces({{V(0), V(1), V(2), V(3)}});
  CHECK(faces == expected);
}

TEST_CASE("Single edge split (fast path)", "[face_cutter]") {
  test_geometry geo{{{0, 0, 0},
                     {200, 0, 0},
                     {200, 200, 0},
                     {0, 200, 0},
                     {0, 100, 0},
                     {200, 100, 0}},
                    {0, 1, 2, 3}};
  auto faces = run(geo, {V(0), V(1), C(5), V(2), V(3), C(4)},
                   {{0, 1, 0, 0, 4, 5, 0}});
  auto expected = canonical_faces({
      {V(0), V(1), C(5), C(4)},
      {C(5), V(2), V(3), C(4)},
  });
  CHECK(faces == expected);
}

TEST_CASE("Single edge interior (cut)", "[face_cutter]") {
  test_geometry geo{{{0, 0, 0},
                     {200, 0, 0},
                     {200, 200, 0},
                     {0, 200, 0},
                     {100, 0, 0},
                     {100, 100, 0}},
                    {0, 1, 2, 3}};
  auto faces = run(geo, {V(0), C(4), V(1), V(2), V(3)},
                   {{0, 1, 0, 0, 4, 5, 0}});
  auto expected = canonical_faces({
      {V(0), C(4), C(5), C(4), V(1), V(2), V(3)},
  });
  CHECK(faces == expected);
}

TEST_CASE("Two parallel edges (3 strips)", "[face_cutter]") {
  test_geometry geo{{{0, 0, 0},
                     {200, 0, 0},
                     {200, 200, 0},
                     {0, 200, 0},
                     {0, 60, 0},
                     {200, 60, 0},
                     {0, 140, 0},
                     {200, 140, 0}},
                    {0, 1, 2, 3}};
  auto faces = run(geo, {V(0), V(1), C(5), C(7), V(2), V(3), C(6), C(4)},
                   {{0, 1, 0, 0, 4, 5, 0}, {0, 2, 0, 0, 6, 7, 1}});
  auto expected = canonical_faces({
      {V(0), V(1), C(5), C(4)},
      {C(4), C(5), C(7), C(6)},
      {C(6), C(7), V(2), V(3)},
  });
  CHECK(faces == expected);
}

TEST_CASE("Interior loop (hole)", "[face_cutter]") {
  test_geometry geo{{{0, 0, 0},
                     {300, 0, 0},
                     {300, 300, 0},
                     {0, 300, 0},
                     {80, 200, 0},
                     {220, 200, 0},
                     {220, 100, 0},
                     {80, 100, 0}},
                    {0, 1, 2, 3}};
  auto faces = run(geo, {V(0), V(1), V(2), V(3)},
                   {{0, 1, 0, 0, 4, 5, 0},
                    {0, 1, 0, 0, 5, 6, 1},
                    {0, 1, 0, 0, 6, 7, 2},
                    {0, 1, 0, 0, 7, 4, 3}});
  auto expected = canonical_faces({
      {V(0), V(1), V(2), V(3), C(4), C(5), C(6), C(7), C(4), V(3)},
      {C(7), C(6), C(5), C(4)},
  });
  CHECK(faces == expected);
}

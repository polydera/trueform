/**
 * @file test_face_cuts.cpp
 * @brief Tests for tf::face_cuts
 *
 * Verifies end-to-end face cutting: intersection_graph → face_cuts.
 * Checks face counts per tag, not vertex ordering.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/views/mapped_range.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <trueform/cut/face_cuts.hpp>
#include <trueform/intersect/exact/intersections_between_polygons.hpp>
#include <trueform/intersect/graph/intersection_graph.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/topology/make_face_membership.hpp>
#include <trueform/topology/make_manifold_edge_link.hpp>

using Index = int;

namespace {

template <int N>
auto make_mesh(const std::vector<std::array<float, 3>> &pts,
               const std::vector<std::array<int, N>> &faces)
    -> tf::polygons_buffer<int, float, 3, N> {
  tf::polygons_buffer<int, float, 3, N> mesh;
  mesh.points_buffer().allocate(pts.size());
  mesh.faces_buffer().allocate(faces.size());
  for (std::size_t i = 0; i < pts.size(); ++i) {
    auto p = mesh.points()[i];
    for (int d = 0; d < 3; ++d)
      p[d] = pts[i][d];
  }
  for (std::size_t i = 0; i < faces.size(); ++i) {
    auto f = mesh.faces()[i];
    for (int d = 0; d < N; ++d)
      f[d] = faces[i][d];
  }
  return mesh;
}

template <int N>
auto build_and_cut(
    std::array<tf::polygons_buffer<int, float, 3, 4>, N> &meshes,
    tf::intersection_graph<Index> &ig,
    tf::face_cuts<Index> &fc) {
  tf::aabb_tree<int, float, 3> trees[N];
  tf::face_membership<int> fms[N];
  tf::manifold_edge_link<int, 4> mels[N];
  for (int i = 0; i < N; ++i) {
    trees[i] = tf::aabb_tree<int, float, 3>(meshes[i].polygons(),
                                             tf::config_tree(4, 4));
    fms[i].build(meshes[i].polygons());
    mels[i] = tf::make_manifold_edge_link(meshes[i].polygons());
  }
  auto forms = tf::make_mapped_range(tf::make_sequence_range(N), [&](int i) {
    return meshes[i].polygons() | tf::tag(trees[i]) | tf::tag(fms[i]) |
           tf::tag(mels[i]);
  });

  tf::exact::intersections_between_polygons<Index, float> ibp;
  ibp.build(forms, tf::intersect_mode::primitives);

  auto &conv = ibp.converter();
  auto get_face = [&](int tag, int object) {
    return forms[tag].faces()[object];
  };
  auto get_mesh_point = [&](int tag, int id) -> tf::point<int32_t, 3> {
    return conv.convert(forms[tag].points()[id]);
  };

  ig.build(ibp, get_face, get_mesh_point);
  fc.build(ig, get_face, get_mesh_point);
}

auto count_per_tag(const tf::face_cuts<Index> &fc, int n_tags)
    -> std::vector<int> {
  std::vector<int> counts(n_tags, 0);
  for (auto desc : fc.descriptors())
    if (desc.tag >= 0 && desc.tag < n_tags)
      ++counts[desc.tag];
  return counts;
}

} // namespace

TEST_CASE("Single cutter through face", "[face_cuts]") {
  auto face = make_mesh<4>(
      {{{-3,-3,0}},{{3,-3,0}},{{3,3,0}},{{-3,3,0}}}, {{{0,1,2,3}}});
  auto cutter = make_mesh<4>(
      {{{-2,0,-1}},{{2,0,-1}},{{2,0,1}},{{-2,0,1}}}, {{{0,1,2,3}}});
  std::array meshes = {face, cutter};

  tf::intersection_graph<Index> ig;
  tf::face_cuts<Index> fc;
  build_and_cut<2>(meshes, ig, fc);

  auto counts = count_per_tag(fc, 2);
  CHECK(counts[0] == 1);  // face: cut (edge interior)
  CHECK(counts[1] == 2);  // cutter: split into 2

  // All output faces non-empty
  for (auto loop : fc.loops())
    CHECK(loop.size() >= 3);
}

TEST_CASE("Two parallel cutters (3 strips)", "[face_cuts]") {
  auto face = make_mesh<4>(
      {{{-3,-3,0}},{{3,-3,0}},{{3,3,0}},{{-3,3,0}}}, {{{0,1,2,3}}});
  auto cutA = make_mesh<4>(
      {{{-4,-1,-1}},{{4,-1,-1}},{{4,-1,1}},{{-4,-1,1}}}, {{{0,1,2,3}}});
  auto cutB = make_mesh<4>(
      {{{-4,1,-1}},{{4,1,-1}},{{4,1,1}},{{-4,1,1}}}, {{{0,1,2,3}}});
  std::array meshes = {face, cutA, cutB};

  tf::intersection_graph<Index> ig;
  tf::face_cuts<Index> fc;
  build_and_cut<3>(meshes, ig, fc);

  auto counts = count_per_tag(fc, 3);
  CHECK(counts[0] == 3);  // face: 3 strips
  CHECK(counts[1] == 1);  // cutA: cut (edge interior)
  CHECK(counts[2] == 1);  // cutB: cut (edge interior)

  for (auto loop : fc.loops())
    CHECK(loop.size() >= 3);
}

TEST_CASE("Perpendicular cutters (X crossing)", "[face_cuts]") {
  auto face = make_mesh<4>(
      {{{-3,-3,0}},{{3,-3,0}},{{3,3,0}},{{-3,3,0}}}, {{{0,1,2,3}}});
  auto cutA = make_mesh<4>(
      {{{-4,0,-1}},{{4,0,-1}},{{4,0,1}},{{-4,0,1}}}, {{{0,1,2,3}}});
  auto cutB = make_mesh<4>(
      {{{0,-4,-1}},{{0,4,-1}},{{0,4,1}},{{0,-4,1}}}, {{{0,1,2,3}}});
  std::array meshes = {face, cutA, cutB};

  tf::intersection_graph<Index> ig;
  tf::face_cuts<Index> fc;
  build_and_cut<3>(meshes, ig, fc);

  auto counts = count_per_tag(fc, 3);
  CHECK(counts[0] == 4);  // face: 4 quadrants
  CHECK(counts[1] == 2);  // cutA: split by cutB
  CHECK(counts[2] == 2);  // cutB: split by cutA

  for (auto loop : fc.loops())
    CHECK(loop.size() >= 3);
}

TEST_CASE("Non-intersecting meshes (empty)", "[face_cuts]") {
  auto A = make_mesh<4>(
      {{{-1,-1,0}},{{1,-1,0}},{{1,1,0}},{{-1,1,0}}}, {{{0,1,2,3}}});
  auto B = make_mesh<4>(
      {{{10,10,0}},{{12,10,0}},{{12,12,0}},{{10,12,0}}}, {{{0,1,2,3}}});
  std::array meshes = {A, B};

  tf::intersection_graph<Index> ig;
  tf::face_cuts<Index> fc;
  build_and_cut<2>(meshes, ig, fc);

  CHECK(fc.loops().size() == 0);
  CHECK(fc.descriptors().size() == 0);
}

/**
 * @file test_cut_graph.cpp
 * @brief Tests for tf::cut_graph
 *
 * Verifies per-edge connectivity, coplanar pair detection, and
 * intersection edge extraction across various mesh configurations.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/views/mapped_range.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <trueform/cut/cut_graph.hpp>
#include <trueform/cut/face_cuts.hpp>
#include <trueform/intersect/exact/intersections_between_polygons.hpp>
#include <trueform/intersect/graph/intersection_graph.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/topology/make_face_membership.hpp>
#include <trueform/topology/make_manifold_edge_link.hpp>

using Index = int;

namespace {

auto make_mesh(const std::vector<std::array<float, 3>> &pts,
               const std::vector<std::array<int, 4>> &faces)
    -> tf::polygons_buffer<int, float, 3, 4> {
  tf::polygons_buffer<int, float, 3, 4> mesh;
  mesh.points_buffer().allocate(pts.size());
  mesh.faces_buffer().allocate(faces.size());
  for (std::size_t i = 0; i < pts.size(); ++i) {
    auto p = mesh.points()[i];
    for (int d = 0; d < 3; ++d)
      p[d] = pts[i][d];
  }
  for (std::size_t i = 0; i < faces.size(); ++i) {
    auto f = mesh.faces()[i];
    for (int d = 0; d < 4; ++d)
      f[d] = faces[i][d];
  }
  return mesh;
}

template <int N>
void run_pipeline(
    std::array<tf::polygons_buffer<int, float, 3, 4>, N> &meshes,
    tf::intersection_graph<Index> &ig,
    tf::face_cuts<Index> &fc,
    tf::cut_graph<Index> &cg) {
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
  cg.build(fc, int(ig.points().size()));
}

auto sorted_edges(const tf::cut_graph<Index> &cg)
    -> std::vector<std::array<Index, 2>> {
  auto ie = cg.intersection_edges();
  std::vector<std::array<Index, 2>> result;
  for (std::size_t i = 0; i < ie.size(); ++i)
    result.push_back(ie[i]);
  std::sort(result.begin(), result.end());
  return result;
}

template <typename Pairs>
auto sorted_coplanar(const Pairs &pairs)
    -> std::vector<std::array<Index, 3>> {
  std::vector<std::array<Index, 3>> result;
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    auto a = std::min(pairs[i].loop_a, pairs[i].loop_b);
    auto b = std::max(pairs[i].loop_a, pairs[i].loop_b);
    result.push_back({a, b, pairs[i].opposing ? 1 : 0});
  }
  std::sort(result.begin(), result.end());
  return result;
}

auto make_cube(float x0, float y0, float z0, float x1, float y1, float z1) {
  return make_mesh(
      {{{x0,y0,z0}}, {{x1,y0,z0}}, {{x1,y1,z0}}, {{x0,y1,z0}},
       {{x0,y0,z1}}, {{x1,y0,z1}}, {{x1,y1,z1}}, {{x0,y1,z1}}},
      {{{3,2,1,0}}, {{4,5,6,7}}, {{0,1,5,4}},
       {{2,3,7,6}}, {{3,0,4,7}}, {{1,2,6,5}}});
}

} // namespace

TEST_CASE("Stacked cubes same size", "[cut_graph]") {
  auto cube0 = make_cube(0, 0, 0, 1, 1, 1);
  auto cube1 = make_cube(0, 0, 1, 1, 1, 2);

  std::array meshes = {cube0, cube1};
  tf::intersection_graph<Index> ig;
  tf::face_cuts<Index> fc;
  tf::cut_graph<Index> cg;
  run_pipeline<2>(meshes, ig, fc, cg);

  CHECK(fc.loops().size() == 10);

  auto pairs = cg.coplanar_pairs();
  CHECK(pairs.size() == 1);
  auto cp = sorted_coplanar(pairs);
  CHECK(cp[0] == std::array<Index, 3>{0, 5, 1}); // opposing


  auto ie = sorted_edges(cg);
  CHECK(ie.size() == 4);
  CHECK(ie == std::vector<std::array<Index, 2>>{{0,1},{0,3},{1,2},{2,3}});
}

TEST_CASE("Small cube on large cube", "[cut_graph]") {
  auto cube0 = make_cube(-2, -2, 0, 2, 2, 2);
  auto cube1 = make_cube(-1, -1, 2, 1, 1, 3);

  std::array meshes = {cube0, cube1};
  tf::intersection_graph<Index> ig;
  tf::face_cuts<Index> fc;
  tf::cut_graph<Index> cg;
  run_pipeline<2>(meshes, ig, fc, cg);

  CHECK(fc.loops().size() == 7);

  auto pairs = cg.coplanar_pairs();
  CHECK(pairs.size() == 1);
  auto cp = sorted_coplanar(pairs);
  CHECK(cp[0] == std::array<Index, 3>{1, 2, 1}); // opposing


  auto ie = sorted_edges(cg);
  CHECK(ie.size() == 4);
  CHECK(ie == std::vector<std::array<Index, 2>>{{0,1},{0,3},{1,2},{2,3}});
}

TEST_CASE("Long box over short box", "[cut_graph]") {
  auto mesh0 = make_cube(0, 0, 0, 2, 1, 1);
  auto mesh1 = make_cube(-0.5f, 0, 1, 2.5f, 1, 2);

  std::array meshes = {mesh0, mesh1};
  tf::intersection_graph<Index> ig;
  tf::face_cuts<Index> fc;
  tf::cut_graph<Index> cg;
  run_pipeline<2>(meshes, ig, fc, cg);

  CHECK(fc.loops().size() == 10);

  auto pairs = cg.coplanar_pairs();
  CHECK(pairs.size() == 1);


  auto ie = sorted_edges(cg);
  CHECK(ie.size() == 4);
  CHECK(ie == std::vector<std::array<Index, 2>>{{0,1},{0,3},{1,2},{2,3}});
}

TEST_CASE("Coplanar quad grids 2x1 over 3x1", "[cut_graph]") {
  auto mesh0 = make_mesh(
      {{{0,0,0}}, {{1,0,0}}, {{2,0,0}}, {{3,0,0}},
       {{0,1,0}}, {{1,1,0}}, {{2,1,0}}, {{3,1,0}}},
      {{{0,1,5,4}}, {{1,2,6,5}}, {{2,3,7,6}}});

  auto mesh1 = make_mesh(
      {{{0.5f,0,0}}, {{1.5f,0,0}}, {{2.5f,0,0}},
       {{0.5f,1,0}}, {{1.5f,1,0}}, {{2.5f,1,0}}},
      {{{0,1,4,3}}, {{1,2,5,4}}});

  std::array meshes = {mesh0, mesh1};
  tf::intersection_graph<Index> ig;
  tf::face_cuts<Index> fc;
  tf::cut_graph<Index> cg;
  run_pipeline<2>(meshes, ig, fc, cg);

  CHECK(fc.loops().size() == 10);

  auto pairs = cg.coplanar_pairs();
  CHECK(pairs.size() == 4);


  auto ie = sorted_edges(cg);
  CHECK(ie.size() == 2);
  CHECK(ie == std::vector<std::array<Index, 2>>{{4,7},{6,9}});
}

TEST_CASE("Tube on quad (2 quads high)", "[cut_graph]") {
  auto mesh0 = make_mesh(
      {{{-2,-2,0}}, {{2,-2,0}}, {{2,2,0}}, {{-2,2,0}}},
      {{{0,1,2,3}}});

  auto mesh1 = make_mesh(
      {{{-1,-1,-1}}, {{1,-1,-1}}, {{1,1,-1}}, {{-1,1,-1}},
       {{-1,-1, 0}}, {{1,-1, 0}}, {{1,1, 0}}, {{-1,1, 0}},
       {{-1,-1, 1}}, {{1,-1, 1}}, {{1,1, 1}}, {{-1,1, 1}}},
      {{{0,1,5,4}}, {{4,5,9,8}}, {{1,2,6,5}}, {{5,6,10,9}},
       {{2,3,7,6}}, {{6,7,11,10}}, {{3,0,4,7}}, {{7,4,8,11}}});

  std::array meshes = {mesh0, mesh1};
  tf::intersection_graph<Index> ig;
  tf::face_cuts<Index> fc;
  tf::cut_graph<Index> cg;
  run_pipeline<2>(meshes, ig, fc, cg);

  CHECK(fc.loops().size() == 10);
  CHECK(cg.coplanar_pairs().size() == 0);


  auto ie = sorted_edges(cg);
  CHECK(ie.size() == 4);
  CHECK(ie == std::vector<std::array<Index, 2>>{{0,1},{0,3},{1,2},{2,3}});
}

TEST_CASE("Split tube on quad (3 meshes)", "[cut_graph]") {
  auto mesh0 = make_mesh(
      {{{-2,-2,0}}, {{2,-2,0}}, {{2,2,0}}, {{-2,2,0}}},
      {{{0,1,2,3}}});

  auto mesh1 = make_mesh(
      {{{-1,-1,0}}, {{1,-1,0}}, {{1,1,0}}, {{-1,1,0}},
       {{-1,-1,1}}, {{1,-1,1}}, {{1,1,1}}, {{-1,1,1}}},
      {{{0,1,5,4}}, {{1,2,6,5}}, {{2,3,7,6}}, {{3,0,4,7}}});

  auto mesh2 = make_mesh(
      {{{-1,-1,-1}}, {{1,-1,-1}}, {{1,1,-1}}, {{-1,1,-1}},
       {{-1,-1, 0}}, {{1,-1, 0}}, {{1,1, 0}}, {{-1,1, 0}}},
      {{{0,1,5,4}}, {{1,2,6,5}}, {{2,3,7,6}}, {{3,0,4,7}}});

  std::array meshes = {mesh0, mesh1, mesh2};
  tf::intersection_graph<Index> ig;
  tf::face_cuts<Index> fc;
  tf::cut_graph<Index> cg;
  run_pipeline<3>(meshes, ig, fc, cg);

  CHECK(fc.loops().size() == 10);
  CHECK(cg.coplanar_pairs().size() == 0);


  auto ie = sorted_edges(cg);
  CHECK(ie.size() == 4);
  CHECK(ie == std::vector<std::array<Index, 2>>{{0,1},{0,3},{1,2},{2,3}});
}

TEST_CASE("Tube penetrating quad", "[cut_graph]") {
  auto mesh0 = make_mesh(
      {{{-2,-2,0}}, {{2,-2,0}}, {{2,2,0}}, {{-2,2,0}}},
      {{{0,1,2,3}}});

  auto mesh1 = make_mesh(
      {{{-1,-1,-1}}, {{1,-1,-1}}, {{1,1,-1}}, {{-1,1,-1}},
       {{-1,-1, 1}}, {{1,-1, 1}}, {{1,1, 1}}, {{-1,1, 1}}},
      {{{0,1,5,4}}, {{1,2,6,5}}, {{2,3,7,6}}, {{3,0,4,7}}});

  std::array meshes = {mesh0, mesh1};
  tf::intersection_graph<Index> ig;
  tf::face_cuts<Index> fc;
  tf::cut_graph<Index> cg;
  run_pipeline<2>(meshes, ig, fc, cg);

  CHECK(fc.loops().size() == 10);
  CHECK(cg.coplanar_pairs().size() == 0);


  auto ie = sorted_edges(cg);
  CHECK(ie.size() == 4);
  CHECK(ie == std::vector<std::array<Index, 2>>{{0,1},{0,2},{1,3},{2,3}});
}

/**
 * @file test_cut_graph_self.cpp
 * @brief Tests for cut_graph self-intersection overload
 *
 * Compares cut_graph built via between-mesh pipeline vs within-mesh pipeline
 * on concatenated meshes. Verifies loops, coplanar pairs, and intersection
 * edges match.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/views/mapped_range.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <trueform/cut/cut_graph.hpp>
#include <trueform/cut/face_cuts.hpp>
#include <trueform/intersect/intersections_between_polygons.hpp>
#include <trueform/intersect/intersections_within_polygons.hpp>
#include <trueform/intersect/graph/intersection_graph.hpp>
#include <trueform/reindex/concatenated.hpp>
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

auto make_cube(float x0, float y0, float z0, float x1, float y1, float z1) {
  return make_mesh(
      {{{x0,y0,z0}}, {{x1,y0,z0}}, {{x1,y1,z0}}, {{x0,y1,z0}},
       {{x0,y0,z1}}, {{x1,y0,z1}}, {{x1,y1,z1}}, {{x0,y1,z1}}},
      {{{3,2,1,0}}, {{4,5,6,7}}, {{0,1,5,4}},
       {{2,3,7,6}}, {{3,0,4,7}}, {{1,2,6,5}}});
}

template <int N>
auto run_between(std::array<tf::polygons_buffer<int, float, 3, 4>, N> &meshes,
                 tf::intersection_graph<Index> &ig, tf::face_cuts<Index> &fc,
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

  tf::intersections_between_polygons<Index, float> ibp;
  ibp.build(forms, tf::intersect_mode::primitives);

  auto &conv = ibp.converter();
  auto apply_to_face = [&](int tag, int object, const auto &f) {
    f(forms[tag].faces()[object]);
  };
  auto get_mesh_point = [&](int tag, int id) -> tf::point<int32_t, 3> {
    return conv.convert(forms[tag].points()[id]);
  };

  ig.build(ibp, apply_to_face, get_mesh_point);
  fc.build(ig, apply_to_face, get_mesh_point);
  cg.build(fc, int(ig.points().size()));
}

template <int N>
auto run_within(std::array<tf::polygons_buffer<int, float, 3, 4>, N> &meshes,
                tf::intersection_graph<Index> &ig, tf::face_cuts<Index> &fc,
                tf::cut_graph<Index> &cg) {
  auto forms = tf::make_mapped_range(tf::make_sequence_range(N),
                                     [&](int i) { return meshes[i].polygons(); });
  auto m = tf::concatenated(forms);

  tf::aabb_tree<int, float, 3> tree(m.polygons(), tf::config_tree(4, 4));
  auto fm = tf::make_face_membership(m.polygons());
  auto mel = tf::make_manifold_edge_link(m.polygons());
  auto tagged = m.polygons() | tf::tag(tree) | tf::tag(fm) | tf::tag(mel);

  tf::intersections_within_polygons<Index, float> iwp;
  iwp.build(tagged, tf::intersect_mode::primitives);

  auto &conv = iwp.converter();
  auto apply_to_face = [&](int, int object, const auto &f) {
    f(m.faces()[object]);
  };
  auto get_mesh_point = [&](int, int id) -> tf::point<int32_t, 3> {
    return conv.convert(m.points()[id]);
  };

  ig.build(iwp, apply_to_face, get_mesh_point);
  fc.build(ig, apply_to_face, get_mesh_point);
  cg.build(fc, ig, tagged);
}

auto count_ie(const tf::cut_graph<Index> &cg) -> std::size_t {
  return cg.intersection_edges().size();
}

} // namespace

TEST_CASE("Self cut_graph: stacked cubes same size", "[cut_graph][self]") {
  auto cube0 = make_cube(0, 0, 0, 1, 1, 1);
  auto cube1 = make_cube(0, 0, 1, 1, 1, 2);
  std::array meshes = {cube0, cube1};

  tf::intersection_graph<Index> ig_b, ig_w;
  tf::face_cuts<Index> fc_b, fc_w;
  tf::cut_graph<Index> cg_b, cg_w;

  run_between<2>(meshes, ig_b, fc_b, cg_b);
  run_within<2>(meshes, ig_w, fc_w, cg_w);

  REQUIRE(fc_b.loops().size() == fc_w.loops().size());
  REQUIRE(cg_b.coplanar_pairs().size() == cg_w.coplanar_pairs().size());
  REQUIRE(count_ie(cg_b) == count_ie(cg_w));
}

TEST_CASE("Self cut_graph: small cube on large cube", "[cut_graph][self]") {
  auto cube0 = make_cube(-2, -2, 0, 2, 2, 2);
  auto cube1 = make_cube(-1, -1, 2, 1, 1, 3);
  std::array meshes = {cube0, cube1};

  tf::intersection_graph<Index> ig_b, ig_w;
  tf::face_cuts<Index> fc_b, fc_w;
  tf::cut_graph<Index> cg_b, cg_w;

  run_between<2>(meshes, ig_b, fc_b, cg_b);
  run_within<2>(meshes, ig_w, fc_w, cg_w);

  REQUIRE(fc_b.loops().size() == fc_w.loops().size());
  REQUIRE(cg_b.coplanar_pairs().size() == cg_w.coplanar_pairs().size());
  REQUIRE(count_ie(cg_b) == count_ie(cg_w));
}

TEST_CASE("Self cut_graph: long box over short box", "[cut_graph][self]") {
  auto cube0 = make_cube(0, 0, 0, 1, 1, 1);
  auto cube1 = make_cube(-1, 0, 1, 2, 1, 2);
  std::array meshes = {cube0, cube1};

  tf::intersection_graph<Index> ig_b, ig_w;
  tf::face_cuts<Index> fc_b, fc_w;
  tf::cut_graph<Index> cg_b, cg_w;

  run_between<2>(meshes, ig_b, fc_b, cg_b);
  run_within<2>(meshes, ig_w, fc_w, cg_w);

  REQUIRE(fc_b.loops().size() == fc_w.loops().size());
  REQUIRE(cg_b.coplanar_pairs().size() == cg_w.coplanar_pairs().size());
  REQUIRE(count_ie(cg_b) == count_ie(cg_w));
}

TEST_CASE("Self cut_graph: crossing cubes (no coplanar)", "[cut_graph][self]") {
  auto cube0 = make_cube(-1, -1, -1, 1, 1, 1);
  auto cube1 = make_cube(-0.5f, -0.5f, 0.5f, 0.5f, 0.5f, 2);
  std::array meshes = {cube0, cube1};

  tf::intersection_graph<Index> ig_b, ig_w;
  tf::face_cuts<Index> fc_b, fc_w;
  tf::cut_graph<Index> cg_b, cg_w;

  run_between<2>(meshes, ig_b, fc_b, cg_b);
  run_within<2>(meshes, ig_w, fc_w, cg_w);

  REQUIRE(fc_b.loops().size() == fc_w.loops().size());
  REQUIRE(cg_b.coplanar_pairs().size() == 0);
  REQUIRE(cg_w.coplanar_pairs().size() == 0);
  REQUIRE(count_ie(cg_b) == count_ie(cg_w));
}

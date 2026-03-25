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
#include <trueform/core/algorithm/parallel_for_each.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/views/mapped_range.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <trueform/cut/cut_graph.hpp>
#include <trueform/cut/face_cuts.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/intersect/exact/intersections_between_polygons.hpp>
#include <trueform/intersect/graph/intersection_graph.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/topology/connect_edges_to_paths.hpp>
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

// ── Box mesh pipeline helper ──────────────────────────────────────

struct box_result {
  tf::intersection_graph<Index> ig;
  tf::face_cuts<Index> fc;
  tf::cut_graph<Index> cg;
  tf::offset_block_buffer<Index, Index> paths;
};

template <typename F0, typename F1>
auto run_box_pipeline(F0 &box0, F1 &box1) -> box_result {
  tf::aabb_tree<int, float, 3> t0(box0.polygons(), tf::config_tree(4, 4));
  tf::aabb_tree<int, float, 3> t1(box1.polygons(), tf::config_tree(4, 4));
  auto fm0 = tf::make_face_membership(box0.polygons());
  auto fm1 = tf::make_face_membership(box1.polygons());
  auto mel0 = tf::make_manifold_edge_link(box0.polygons());
  auto mel1 = tf::make_manifold_edge_link(box1.polygons());
  auto f0 = box0.polygons() | tf::tag(t0) | tf::tag(fm0) | tf::tag(mel0);
  auto f1 = box1.polygons() | tf::tag(t1) | tf::tag(fm1) | tf::tag(mel1);

  decltype(f0) forms[] = {f0, f1};
  auto range = tf::make_range(forms, forms + 2);

  tf::exact::intersections_between_polygons<Index, float> ibp;
  ibp.build(range, tf::intersect_mode::primitives);

  auto &conv = ibp.converter();
  auto get_face = [&](int tag, int object) {
    return range[tag].faces()[object];
  };
  auto get_mesh_point = [&](int tag, int id) -> tf::point<int32_t, 3> {
    return conv.convert(
        tf::transformed(range[tag].points()[id], tf::frame_of(range[tag])));
  };

  box_result r;
  r.ig.build(ibp, get_face, get_mesh_point);
  r.fc.build(r.ig, get_face, get_mesh_point);
  r.cg.build(r.fc, int(r.ig.points().size()));

  auto ie = r.cg.intersection_edges();
  tf::blocked_buffer<Index, 2> edge_buf;
  edge_buf.allocate(ie.size());
  for (std::size_t i = 0; i < ie.size(); ++i) {
    edge_buf[i][0] = ie[i][0];
    edge_buf[i][1] = ie[i][1];
  }
  r.paths = tf::connect_edges_to_paths(tf::make_edges(edge_buf));
  return r;
}

auto check_all_paths_closed(const box_result &r) {
  for (auto path : r.paths) {
    REQUIRE(path.size() > 2);
    CHECK(path[0] == path[path.size() - 1]);
  }
}

auto check_coplanar_all(const box_result &r, bool expect_opposing) {
  auto pairs = r.cg.coplanar_pairs();
  for (std::size_t i = 0; i < pairs.size(); ++i)
    CHECK(pairs[i].opposing == expect_opposing);
}

// ── Box mesh tests: stacked (opposing coplanar) ───────────────────

TEST_CASE("Stacked boxes same resolution", "[cut_graph][box]") {
  auto box0 = tf::make_box_mesh<int>(2.0f, 2.0f, 2.0f, 4, 4, 4);
  auto box1 = tf::make_box_mesh<int>(2.0f, 2.0f, 2.0f, 4, 4, 4);
  tf::parallel_for_each(box0.polygons().points(), [](auto pt) { pt[1] -= 1.0f; });
  tf::parallel_for_each(box1.polygons().points(), [](auto pt) { pt[1] += 1.0f; });

  auto r = run_box_pipeline(box0, box1);
  CHECK(r.ig.points().size() == 41);
  CHECK(r.cg.intersection_edges().size() == 16);
  CHECK(r.paths.size() == 1);
  check_all_paths_closed(r);
  CHECK(r.cg.coplanar_pairs().size() == 64);
  check_coplanar_all(r, true);
}

TEST_CASE("Stacked boxes different resolution", "[cut_graph][box]") {
  auto box0 = tf::make_box_mesh<int>(2.0f, 2.0f, 2.0f, 3, 3, 3);
  auto box1 = tf::make_box_mesh<int>(2.0f, 2.0f, 2.0f, 5, 5, 5);
  tf::parallel_for_each(box0.polygons().points(), [](auto pt) { pt[1] -= 1.0f; });
  tf::parallel_for_each(box1.polygons().points(), [](auto pt) { pt[1] += 1.0f; });

  auto r = run_box_pipeline(box0, box1);
  CHECK(r.ig.points().size() == 131);
  CHECK(r.cg.intersection_edges().size() == 28);
  CHECK(r.paths.size() == 1);
  check_all_paths_closed(r);
  CHECK(r.cg.coplanar_pairs().size() == 152);
  check_coplanar_all(r, true);
}

TEST_CASE("Wider box low res", "[cut_graph][box]") {
  auto box0 = tf::make_box_mesh<int>(2.0f, 2.0f, 2.0f, 1, 1, 1);
  auto box1 = tf::make_box_mesh<int>(3.0f, 2.0f, 3.0f, 2, 2, 2);
  tf::parallel_for_each(box0.polygons().points(), [](auto pt) { pt[1] -= 1.0f; });
  tf::parallel_for_each(box1.polygons().points(), [](auto pt) { pt[1] += 1.0f; });

  auto r = run_box_pipeline(box0, box1);
  CHECK(r.ig.points().size() == 15);
  CHECK(r.cg.intersection_edges().size() == 12);
  CHECK(r.paths.size() == 1);
  check_all_paths_closed(r);
  CHECK(r.cg.coplanar_pairs().size() == 12);
  check_coplanar_all(r, true);
}

TEST_CASE("Wider box high res", "[cut_graph][box]") {
  auto box0 = tf::make_box_mesh<int>(2.0f, 2.0f, 2.0f, 4, 4, 4);
  auto box1 = tf::make_box_mesh<int>(3.0f, 2.0f, 3.0f, 5, 5, 5);
  tf::parallel_for_each(box0.polygons().points(), [](auto pt) { pt[1] -= 1.0f; });
  tf::parallel_for_each(box1.polygons().points(), [](auto pt) { pt[1] += 1.0f; });

  auto r = run_box_pipeline(box0, box1);
  CHECK(r.ig.points().size() == 159);
  CHECK(r.cg.intersection_edges().size() == 44);
  CHECK(r.paths.size() == 1);
  check_all_paths_closed(r);
  CHECK(r.cg.coplanar_pairs().size() == 172);
  check_coplanar_all(r, true);
}

// ── Box mesh tests: inside (aligned coplanar) ─────────────────────

TEST_CASE("Box inside coplanar top", "[cut_graph][box]") {
  auto box0 = tf::make_box_mesh<int>(3.0f, 2.0f, 3.0f, 3, 3, 3);
  auto box1 = tf::make_box_mesh<int>(1.0f, 2.0f, 1.0f, 2, 2, 2);

  auto r = run_box_pipeline(box0, box1);
  CHECK(r.ig.points().size() == 18);
  CHECK(r.cg.intersection_edges().size() == 16);
  CHECK(r.paths.size() == 2);
  check_all_paths_closed(r);
  CHECK(r.cg.coplanar_pairs().size() == 16);
  check_coplanar_all(r, false);
}

TEST_CASE("Box inside coplanar top high res", "[cut_graph][box]") {
  auto box0 = tf::make_box_mesh<int>(3.0f, 2.0f, 3.0f, 5, 5, 5);
  auto box1 = tf::make_box_mesh<int>(1.0f, 2.0f, 1.0f, 4, 4, 4);

  auto r = run_box_pipeline(box0, box1);
  CHECK(r.ig.points().size() == 150);
  CHECK(r.cg.intersection_edges().size() == 58);
  CHECK(r.paths.size() == 2);
  check_all_paths_closed(r);
  CHECK(r.cg.coplanar_pairs().size() == 159);
  check_coplanar_all(r, false);
}

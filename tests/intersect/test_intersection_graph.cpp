/**
 * @file test_intersection_graph.cpp
 * @brief Tests for tf::intersection_graph
 *
 * Verifies edge extraction, crossing detection (EE, VE, VV), same-t merge,
 * boundary edge skipping, and coplanar face handling.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <set>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/views/mapped_range.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <trueform/core/views/zip.hpp>
#include <trueform/intersect/exact/intersections_between_polygons.hpp>
#include <trueform/intersect/graph/intersection_graph.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/topology/make_face_membership.hpp>
#include <trueform/topology/make_manifold_edge_link.hpp>

using Index = int;
using IG = tf::intersection_graph<int>;

// =============================================================================
// Helpers
// =============================================================================

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
auto build_graph(std::array<tf::polygons_buffer<int, float, 3, 4>, N> &meshes,
                 tf::exact::intersections_between_polygons<Index, float> &ibp,
                 IG &ig,
                 tf::intersect_mode mode = tf::intersect_mode::primitives) {
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

  ibp.build(forms, mode);

  auto &conv = ibp.converter();
  auto ipts = ibp.intersection_points();
  auto get_face = [&](int tag, int object) {
    return forms[tag].faces()[object];
  };
  auto get_point = [&](int tag, int id) -> tf::point<int32_t, 3> {
    if (tag == -1)
      return ipts[id];
    return conv.convert(forms[tag].points()[id]);
  };
  auto get_flat_id = [&](const auto &rec) -> int {
    return ibp.get_flat_index(rec);
  };

  ig.build(ibp.intersections(), get_face, get_point, get_flat_id);
}

auto collect_edge_points(const IG &ig) -> std::set<int> {
  std::set<int> pts;
  for (auto group : ig.edge_groups())
    for (const auto &e : group) {
      pts.insert(e.point_0);
      pts.insert(e.point_1);
    }
  return pts;
}

auto all_groups_have_n_instances(const IG &ig, std::size_t expected) -> bool {
  for (auto group : ig.edge_groups())
    if (group.size() != expected)
      return false;
  return true;
}

auto check_face_edge_tags(
    const IG &ig,
    const tf::exact::intersections_between_polygons<Index, float> &ibp)
    -> bool {
  auto &&edge_data = ig.edge_groups();
  for (auto [subrange, loop, face_edges] :
       tf::zip(ibp.intersections(), ig.loops(), ig.edges())) {
    if (subrange.size() == 0)
      continue;
    auto tag = subrange[0].tag;
    auto object = subrange[0].object;
    for (auto inst_idx : face_edges) {
      int flat = 0;
      for (auto group : edge_data) {
        for (std::size_t m = 0; m < group.size(); ++m) {
          if (flat == inst_idx) {
            auto &e = group[m];
            if (e.tag != tag || e.object != object)
              return false;
            goto next;
          }
          ++flat;
        }
      }
      return false;
      next:;
    }
  }
  return true;
}

// =============================================================================
// Non-coplanar crossing tests
// =============================================================================

TEST_CASE("EE proper crossing (X on face)", "[graph][crossing]") {
  auto face = make_mesh<4>({{{-3,-3,0}},{{3,-3,0}},{{3,3,0}},{{-3,3,0}}}, {{{0,1,2,3}}});
  auto cutA = make_mesh<4>({{{-2,-2,-1}},{{2,2,-1}},{{2,2,1}},{{-2,-2,1}}}, {{{0,1,2,3}}});
  auto cutB = make_mesh<4>({{{-2,2,-1}},{{2,-2,-1}},{{2,-2,1}},{{-2,2,1}}}, {{{0,1,2,3}}});
  std::array meshes = {face, cutA, cutB};

  tf::exact::intersections_between_polygons<Index, float> ibp;
  IG ig;
  build_graph<3>(meshes, ibp, ig);

  CHECK(ibp.intersection_points().size() == 6);
  CHECK(ig.edge_groups().size() == 6);
  CHECK(all_groups_have_n_instances(ig, 2));
  CHECK(collect_edge_points(ig).size() == 7);
  CHECK(ig.edges().begin()[0].size() == 4);
  CHECK(check_face_edge_tags(ig, ibp));
}

TEST_CASE("VE T-junction", "[graph][crossing]") {
  auto face = make_mesh<4>({{{-3,-3,0}},{{3,-3,0}},{{3,3,0}},{{-3,3,0}}}, {{{0,1,2,3}}});
  auto cutA = make_mesh<4>({{{-2,0,-1}},{{2,0,-1}},{{2,0,1}},{{-2,0,1}}}, {{{0,1,2,3}}});
  auto cutB = make_mesh<4>({{{0,-2,-1}},{{0,0,-1}},{{0,0,1}},{{0,-2,1}}}, {{{0,1,2,3}}});
  std::array meshes = {face, cutA, cutB};

  tf::exact::intersections_between_polygons<Index, float> ibp;
  IG ig;
  build_graph<3>(meshes, ibp, ig);

  CHECK(ibp.intersection_points().size() == 6);
  CHECK(ig.edge_groups().size() == 5);

  auto pts = collect_edge_points(ig);
  auto cb = ig.crossing_base();
  auto n_crossing =
      std::count_if(pts.begin(), pts.end(), [cb](int id) { return id >= cb; });
  CHECK(n_crossing == 0);
  CHECK(ig.edges().begin()[0].size() == 3);
  CHECK(check_face_edge_tags(ig, ibp));
}

TEST_CASE("VV same-position endpoints", "[graph][crossing]") {
  auto face = make_mesh<4>({{{-3,-3,0}},{{3,-3,0}},{{3,3,0}},{{-3,3,0}}}, {{{0,1,2,3}}});
  auto cutA = make_mesh<4>({{{-2,0,-1}},{{0,0,-1}},{{0,0,1}},{{-2,0,1}}}, {{{0,1,2,3}}});
  auto cutB = make_mesh<4>({{{0,-2,-1}},{{0,0,-1}},{{0,0,1}},{{0,-2,1}}}, {{{0,1,2,3}}});
  std::array meshes = {face, cutA, cutB};

  tf::exact::intersections_between_polygons<Index, float> ibp;
  IG ig;
  build_graph<3>(meshes, ibp, ig);

  CHECK(ibp.intersection_points().size() == 6);
  CHECK(ig.point_remap().size() > 0);
  CHECK(ig.edge_groups().size() == 2);
  CHECK(all_groups_have_n_instances(ig, 2));
  CHECK(collect_edge_points(ig).size() == 3);
  CHECK(ig.edges().begin()[0].size() == 2);
  CHECK(check_face_edge_tags(ig, ibp));
}

TEST_CASE("Multiple EE (3 crossings at origin)", "[graph][crossing]") {
  auto face = make_mesh<4>({{{-4,-4,0}},{{4,-4,0}},{{4,4,0}},{{-4,4,0}}}, {{{0,1,2,3}}});
  auto cutA = make_mesh<4>({{{-3,0,-1}},{{3,0,-1}},{{3,0,1}},{{-3,0,1}}}, {{{0,1,2,3}}});
  auto cutB = make_mesh<4>({{{-3,-3,-1}},{{3,3,-1}},{{3,3,1}},{{-3,-3,1}}}, {{{0,1,2,3}}});
  auto cutC = make_mesh<4>({{{-3,3,-1}},{{3,-3,-1}},{{3,-3,1}},{{-3,3,1}}}, {{{0,1,2,3}}});
  std::array meshes = {face, cutA, cutB, cutC};

  tf::exact::intersections_between_polygons<Index, float> ibp;
  IG ig;
  build_graph<4>(meshes, ibp, ig);

  CHECK(ibp.intersection_points().size() == 12);
  CHECK(ig.point_remap().size() > 0);
  CHECK(ig.edge_groups().size() == 8);

  auto pts = collect_edge_points(ig);
  auto cb = ig.crossing_base();
  auto n_crossing =
      std::count_if(pts.begin(), pts.end(), [cb](int id) { return id >= cb; });
  CHECK(n_crossing == 1);
  CHECK(check_face_edge_tags(ig, ibp));
}

TEST_CASE("No crossings (parallel cutters)", "[graph][crossing]") {
  auto face = make_mesh<4>({{{-3,-3,0}},{{3,-3,0}},{{3,3,0}},{{-3,3,0}}}, {{{0,1,2,3}}});
  auto cutA = make_mesh<4>({{{-2,-1,-1}},{{2,-1,-1}},{{2,-1,1}},{{-2,-1,1}}}, {{{0,1,2,3}}});
  auto cutB = make_mesh<4>({{{-2,1,-1}},{{2,1,-1}},{{2,1,1}},{{-2,1,1}}}, {{{0,1,2,3}}});
  std::array meshes = {face, cutA, cutB};

  tf::exact::intersections_between_polygons<Index, float> ibp;
  IG ig;
  build_graph<3>(meshes, ibp, ig);

  CHECK(ibp.intersection_points().size() == 4);
  CHECK(ig.edge_groups().size() == 2);
  CHECK(all_groups_have_n_instances(ig, 2));
  CHECK(ig.point_remap().size() == 0);
  CHECK(ig.crossing_points().size() == 0);
  CHECK(collect_edge_points(ig).size() == 4);
  CHECK(ig.edges().begin()[0].size() == 2);
  CHECK(check_face_edge_tags(ig, ibp));
}

TEST_CASE("EE with parallel splitters", "[graph][crossing]") {
  auto face = make_mesh<4>({{{-4,-4,0}},{{4,-4,0}},{{4,4,0}},{{-4,4,0}}}, {{{0,1,2,3}}});
  auto cutter = make_mesh<4>({{{-3,0,-1}},{{3,0,-1}},{{3,0,1}},{{-3,0,1}}}, {{{0,1,2,3}}});
  auto splitA = make_mesh<4>({{{-1,-2,-1}},{{-1,2,-1}},{{-1,2,1}},{{-1,-2,1}}}, {{{0,1,2,3}}});
  auto splitB = make_mesh<4>({{{1,-2,-1}},{{1,2,-1}},{{1,2,1}},{{1,-2,1}}}, {{{0,1,2,3}}});
  std::array meshes = {face, cutter, splitA, splitB};

  tf::exact::intersections_between_polygons<Index, float> ibp;
  IG ig;
  build_graph<4>(meshes, ibp, ig);

  CHECK(ibp.intersection_points().size() == 10);
  CHECK(ig.edge_groups().size() == 11);
  CHECK(all_groups_have_n_instances(ig, 2));
  CHECK(ig.crossing_points().size() == 2);
  CHECK(ig.edges().begin()[0].size() == 7);
  CHECK(check_face_edge_tags(ig, ibp));
}

// =============================================================================
// Coplanar tests
// =============================================================================

TEST_CASE("Coplanar partial overlap", "[graph][coplanar]") {
  auto A = make_mesh<4>({{{-2,-2,0}},{{2,-2,0}},{{2,2,0}},{{-2,2,0}}}, {{{0,1,2,3}}});
  auto B = make_mesh<4>({{{0,-1,0}},{{3,-1,0}},{{3,1,0}},{{0,1,0}}}, {{{0,1,2,3}}});
  std::array meshes = {A, B};

  tf::exact::intersections_between_polygons<Index, float> ibp;
  IG ig;
  build_graph<2>(meshes, ibp, ig);

  CHECK(ibp.intersection_points().size() == 4);
  CHECK(ig.edge_groups().size() == 4);
  CHECK(ig.edges().begin()[0].size() == 3);
  CHECK(check_face_edge_tags(ig, ibp));
}

TEST_CASE("Coplanar partial overlap with splitter", "[graph][coplanar]") {
  auto A = make_mesh<4>({{{-2,-2,0}},{{2,-2,0}},{{2,2,0}},{{-2,2,0}}}, {{{0,1,2,3}}});
  auto B = make_mesh<4>({{{0,-1,0}},{{3,-1,0}},{{3,1,0}},{{0,1,0}}}, {{{0,1,2,3}}});
  auto C = make_mesh<4>({{{1,-3,-1}},{{1,3,-1}},{{1,3,1}},{{1,-3,1}}}, {{{0,1,2,3}}});
  std::array meshes = {A, B, C};

  tf::exact::intersections_between_polygons<Index, float> ibp;
  IG ig;
  build_graph<3>(meshes, ibp, ig);

  CHECK(ibp.intersection_points().size() == 8);
  CHECK(ig.edge_groups().size() == 9);
  CHECK(ig.edges().begin()[0].size() == 8);
  CHECK(check_face_edge_tags(ig, ibp));
}

TEST_CASE("Coplanar B fully inside A", "[graph][coplanar]") {
  auto A = make_mesh<4>({{{-3,-3,0}},{{3,-3,0}},{{3,3,0}},{{-3,3,0}}}, {{{0,1,2,3}}});
  auto B = make_mesh<4>({{{-1,-1,0}},{{1,-1,0}},{{1,1,0}},{{-1,1,0}}}, {{{0,1,2,3}}});
  std::array meshes = {A, B};

  tf::exact::intersections_between_polygons<Index, float> ibp;
  IG ig;
  build_graph<2>(meshes, ibp, ig);

  CHECK(ibp.intersection_points().size() == 4);
  CHECK(ig.edge_groups().size() == 4);
  CHECK(ig.edges().begin()[0].size() == 4);
  CHECK(ig.edges().begin()[1].size() == 0);
  CHECK(check_face_edge_tags(ig, ibp));
}

TEST_CASE("Coplanar B inside A, shared bottom edge", "[graph][coplanar]") {
  auto A = make_mesh<4>({{{-3,-3,0}},{{3,-3,0}},{{3,3,0}},{{-3,3,0}}}, {{{0,1,2,3}}});
  auto B = make_mesh<4>({{{-1,-3,0}},{{1,-3,0}},{{1,0,0}},{{-1,0,0}}}, {{{0,1,2,3}}});
  std::array meshes = {A, B};

  tf::exact::intersections_between_polygons<Index, float> ibp;
  IG ig;
  build_graph<2>(meshes, ibp, ig);

  CHECK(ibp.intersection_points().size() == 4);
  CHECK(ig.edge_groups().size() == 3);
  CHECK(ig.edges().begin()[0].size() == 3);
  CHECK(ig.edges().begin()[1].size() == 0);
  CHECK(check_face_edge_tags(ig, ibp));
}

TEST_CASE("Coplanar identical quads (no edges)", "[graph][coplanar]") {
  auto A = make_mesh<4>({{{-2,-2,0}},{{2,-2,0}},{{2,2,0}},{{-2,2,0}}}, {{{0,1,2,3}}});
  auto B = make_mesh<4>({{{-2,-2,0}},{{2,-2,0}},{{2,2,0}},{{-2,2,0}}}, {{{0,1,2,3}}});
  std::array meshes = {A, B};

  tf::exact::intersections_between_polygons<Index, float> ibp;
  IG ig;
  build_graph<2>(meshes, ibp, ig);

  CHECK(ibp.intersection_points().size() == 4);
  CHECK(ig.edge_groups().size() == 0);
  CHECK(ig.edges().begin()[0].size() == 0);
  CHECK(ig.edges().begin()[1].size() == 0);
}

TEST_CASE("Coplanar corner overlap", "[graph][coplanar]") {
  auto A = make_mesh<4>({{{-2,-2,0}},{{2,-2,0}},{{2,2,0}},{{-2,2,0}}}, {{{0,1,2,3}}});
  auto B = make_mesh<4>({{{1,1,0}},{{4,1,0}},{{4,4,0}},{{1,4,0}}}, {{{0,1,2,3}}});
  std::array meshes = {A, B};

  tf::exact::intersections_between_polygons<Index, float> ibp;
  IG ig;
  build_graph<2>(meshes, ibp, ig);

  CHECK(ibp.intersection_points().size() == 4);
  CHECK(ig.edge_groups().size() == 4);
  CHECK(ig.edges().begin()[0].size() == 2);
  CHECK(ig.edges().begin()[1].size() == 2);
  CHECK(check_face_edge_tags(ig, ibp));
}

TEST_CASE("Coplanar shared edge (adjacent quads, no overlap)", "[graph][coplanar]") {
  auto A = make_mesh<4>({{{0,0,0}},{{2,0,0}},{{2,2,0}},{{0,2,0}}}, {{{0,1,2,3}}});
  auto B = make_mesh<4>({{{2,0,0}},{{4,0,0}},{{4,2,0}},{{2,2,0}}}, {{{0,1,2,3}}});
  std::array meshes = {A, B};

  tf::exact::intersections_between_polygons<Index, float> ibp;
  IG ig;
  build_graph<2>(meshes, ibp, ig);

  CHECK(ibp.intersection_points().size() == 2);
  CHECK(ig.edge_groups().size() == 0);
  CHECK(ig.edges().begin()[0].size() == 0);
  CHECK(ig.edges().begin()[1].size() == 0);
}

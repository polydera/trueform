/**
 * @file test_exact_intersections.cpp
 * @brief Tests for tf::intersections_between_polygons (primitives mode)
 *
 * Verifies 5-type classification (VV, VE, EE, VF, EF) by counting unique
 * intersection points per canonical type. Types are symmetric:
 *   (edge, face) == (face, edge) -> EF
 *   (vertex, edge) == (edge, vertex) -> VE
 *   (vertex, face) == (face, vertex) -> VF
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/intersect/intersections_between_polygons.hpp>
#include <trueform/intersect/intersections_within_polygons.hpp>
#include <trueform/intersect/intersect_mode.hpp>
#include <trueform/reindex/concatenated.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/topology/make_face_membership.hpp>
#include <trueform/topology/make_manifold_edge_link.hpp>
#include <set>

using Index = int;

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

struct type_counts {
  int vv = 0, ve = 0, ee = 0, vf = 0, ef = 0;
};

template <typename MeshA, typename MeshB>
auto count_primitives(const MeshA &mesh_a, const MeshB &mesh_b) -> type_counts {
  tf::aabb_tree<int, float, 3> ta(mesh_a.polygons(), tf::config_tree(4, 4));
  tf::aabb_tree<int, float, 3> tb(mesh_b.polygons(), tf::config_tree(4, 4));
  auto fma = tf::make_face_membership(mesh_a.polygons());
  auto fmb = tf::make_face_membership(mesh_b.polygons());
  auto mela = tf::make_manifold_edge_link(mesh_a.polygons());
  auto melb = tf::make_manifold_edge_link(mesh_b.polygons());
  auto tag_a = mesh_a.polygons() | tf::tag(ta) | tf::tag(fma) | tf::tag(mela);
  auto tag_b = mesh_b.polygons() | tf::tag(tb) | tf::tag(fmb) | tf::tag(melb);

  tf::intersections_between_polygons<Index, float> ibp;
  ibp.build(tag_a, tag_b, tf::intersect_mode::primitives);

  std::set<int> s_vv, s_ve, s_ee, s_vf, s_ef;
  for (auto subrange : ibp.intersections()) {
    for (auto &r : subrange) {
      if (r.tag != 0)
        continue;
      int id = r.id;
      auto a = r.target.label, b = r.target_other.label;
      using T = tf::topo_type;
      if (a == T::vertex && b == T::vertex)
        s_vv.insert(id);
      else if ((a == T::vertex && b == T::edge) ||
               (a == T::edge && b == T::vertex))
        s_ve.insert(id);
      else if (a == T::edge && b == T::edge)
        s_ee.insert(id);
      else if ((a == T::vertex && b == T::face) ||
               (a == T::face && b == T::vertex))
        s_vf.insert(id);
      else if ((a == T::edge && b == T::face) ||
               (a == T::face && b == T::edge))
        s_ef.insert(id);
    }
  }
  return {int(s_vv.size()), int(s_ve.size()), int(s_ee.size()),
          int(s_vf.size()), int(s_ef.size())};
}

template <typename Tagged>
auto count_primitives_n(Tagged *forms, std::size_t n) -> type_counts {
  tf::intersections_between_polygons<Index, float> ibp;
  ibp.build(tf::make_range(forms, forms + n), tf::intersect_mode::primitives);

  std::set<int> s_vv, s_ve, s_ee, s_vf, s_ef;
  for (auto subrange : ibp.intersections()) {
    for (auto &r : subrange) {
      if (r.tag != 0)
        continue;
      int id = r.id;
      auto a = r.target.label, b = r.target_other.label;
      using T = tf::topo_type;
      if (a == T::vertex && b == T::vertex)
        s_vv.insert(id);
      else if ((a == T::vertex && b == T::edge) ||
               (a == T::edge && b == T::vertex))
        s_ve.insert(id);
      else if (a == T::edge && b == T::edge)
        s_ee.insert(id);
      else if ((a == T::vertex && b == T::face) ||
               (a == T::face && b == T::vertex))
        s_vf.insert(id);
      else if ((a == T::edge && b == T::face) ||
               (a == T::face && b == T::edge))
        s_ef.insert(id);
    }
  }
  return {int(s_vv.size()), int(s_ve.size()), int(s_ee.size()),
          int(s_vf.size()), int(s_ef.size())};
}


/// Concatenate two meshes and run self-intersection. Should match between.
template <typename MeshA, typename MeshB>
auto count_self_primitives(const MeshA &mesh_a, const MeshB &mesh_b)
    -> type_counts {
  auto merged = tf::concatenated(mesh_a.polygons(), mesh_b.polygons());
  tf::aabb_tree<int, float, 3> tree(merged.polygons(), tf::config_tree(4, 4));
  auto fm = tf::make_face_membership(merged.polygons());
  auto mel = tf::make_manifold_edge_link(merged.polygons());
  auto tagged = merged.polygons() | tf::tag(tree) | tf::tag(fm) | tf::tag(mel);

  tf::intersections_within_polygons<Index, float> iwp;
  iwp.build(tagged, tf::intersect_mode::primitives);

  std::set<int> s_vv, s_ve, s_ee, s_vf, s_ef;
  for (auto subrange : iwp.intersections()) {
    for (auto &r : subrange) {
      int id = r.id;
      auto a = r.target.label, b = r.target_other.label;
      using T = tf::topo_type;
      if (a == T::vertex && b == T::vertex)
        s_vv.insert(id);
      else if ((a == T::vertex && b == T::edge) ||
               (a == T::edge && b == T::vertex))
        s_ve.insert(id);
      else if (a == T::edge && b == T::edge)
        s_ee.insert(id);
      else if ((a == T::vertex && b == T::face) ||
               (a == T::face && b == T::vertex))
        s_vf.insert(id);
      else if ((a == T::edge && b == T::face) ||
               (a == T::face && b == T::edge))
        s_ef.insert(id);
    }
  }
  return {int(s_vv.size()), int(s_ve.size()), int(s_ee.size()),
          int(s_vf.size()), int(s_ef.size())};
}


#define CHECK_SELF_MATCHES(a, b, c) \
  { auto s = count_self_primitives(a, b); \
    CHECK(s.vv == c.vv); CHECK(s.ve == c.ve); CHECK(s.ee == c.ee); \
    CHECK(s.vf == c.vf); CHECK(s.ef == c.ef); }

/// Concatenate N forms and run self-intersection.
template <typename Tagged>
auto count_self_primitives_n(Tagged *forms, std::size_t n) -> type_counts {
  // Concatenate all forms' underlying polygons
  auto merged = tf::concatenated(tf::make_range(forms, forms + n));
  tf::aabb_tree<int, float, 3> tree(merged.polygons(), tf::config_tree(4, 4));
  auto fm = tf::make_face_membership(merged.polygons());
  auto mel = tf::make_manifold_edge_link(merged.polygons());
  auto tagged = merged.polygons() | tf::tag(tree) | tf::tag(fm) | tf::tag(mel);

  tf::intersections_within_polygons<Index, float> iwp;
  iwp.build(tagged, tf::intersect_mode::primitives);

  std::set<int> s_vv, s_ve, s_ee, s_vf, s_ef;
  for (auto subrange : iwp.intersections()) {
    for (auto &r : subrange) {
      int id = r.id;
      auto a = r.target.label, b = r.target_other.label;
      using T = tf::topo_type;
      if (a == T::vertex && b == T::vertex) s_vv.insert(id);
      else if ((a == T::vertex && b == T::edge) || (a == T::edge && b == T::vertex)) s_ve.insert(id);
      else if (a == T::edge && b == T::edge) s_ee.insert(id);
      else if ((a == T::vertex && b == T::face) || (a == T::face && b == T::vertex)) s_vf.insert(id);
      else if ((a == T::edge && b == T::face) || (a == T::face && b == T::edge)) s_ef.insert(id);
    }
  }
  return {int(s_vv.size()), int(s_ve.size()), int(s_ee.size()),
          int(s_vf.size()), int(s_ef.size())};
}

#define CHECK_SELF_MATCHES_N(forms, n, c) \
  { auto s = count_self_primitives_n(forms, n); \
    CHECK(s.vv == c.vv); CHECK(s.ve == c.ve); CHECK(s.ee == c.ee); \
    CHECK(s.vf == c.vf); CHECK(s.ef == c.ef); }

// =============================================================================
// Coplanar type isolation
// =============================================================================

TEST_CASE("Coplanar VV only (identical quads)", "[exact][coplanar]") {
  auto a = make_mesh<4>(
      {{{-1, -1, 0}}, {{1, -1, 0}}, {{1, 1, 0}}, {{-1, 1, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{-1, -1, 0}}, {{1, -1, 0}}, {{1, 1, 0}}, {{-1, 1, 0}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 4);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Coplanar VE only (B vertices on A edges)", "[exact][coplanar]") {
  auto a = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 0, 0}}, {{4, 2, 0}}, {{0, 2, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{1, 0, 0}}, {{3, 0, 0}}, {{3, 2, 0}}, {{1, 2, 0}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 4);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Coplanar EE only (crossing strips)", "[exact][coplanar]") {
  auto a = make_mesh<4>(
      {{{-2, -1, 0}}, {{2, -1, 0}}, {{2, 1, 0}}, {{-2, 1, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{-1, -2, 0}}, {{1, -2, 0}}, {{1, 2, 0}}, {{-1, 2, 0}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 4);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Coplanar VF only (B inside A)", "[exact][coplanar]") {
  auto a = make_mesh<4>(
      {{{-3, -3, 0}}, {{3, -3, 0}}, {{3, 3, 0}}, {{-3, 3, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{-1, -1, 0}}, {{1, -1, 0}}, {{1, 1, 0}}, {{-1, 1, 0}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 4);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Coplanar VV+VE mixed", "[exact][coplanar]") {
  auto a = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 0, 0}}, {{4, 2, 0}}, {{0, 2, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{0, 0, 0}}, {{2, 0, 0}}, {{2, 2, 0}}, {{0, 2, 0}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 2);
  CHECK(c.ve == 2);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Coplanar EE+VF mixed (partial overlap)", "[exact][coplanar]") {
  auto a = make_mesh<4>(
      {{{-2, -2, 0}}, {{2, -2, 0}}, {{2, 2, 0}}, {{-2, 2, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{-1, -1, 0}}, {{3, -1, 0}}, {{3, 3, 0}}, {{-1, 3, 0}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 2);
  CHECK(c.vf == 2);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

// =============================================================================
// Non-coplanar: all EF (fast path - all vertices off opposite plane)
// =============================================================================

TEST_CASE("Crossing quads (all EF)", "[exact][ef]") {
  auto a = make_mesh<4>(
      {{{-5, -5, 0}}, {{5, -5, 0}}, {{5, 5, 0}}, {{-5, 5, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{-2, 0, -2}}, {{2, 0, -2}}, {{2, 0, 2}}, {{-2, 0, 2}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 2);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Crossing triangles (all EF)", "[exact][ef]") {
  auto a = make_mesh<3>(
      {{{-5, -5, 0}}, {{5, -5, 0}}, {{0, 5, 0}}}, {{{0, 1, 2}}});
  auto b = make_mesh<3>(
      {{{-2, 0, -2}}, {{2, 0, -2}}, {{0, 0, 2}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 2);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Edge hits face interior (1 EF + 1 VF)", "[exact][ef]") {
  auto a = make_mesh<4>(
      {{{-5, -5, 0}}, {{5, -5, 0}}, {{5, 5, 0}}, {{-5, 5, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{0, 0, 2}}, {{0, 0, -2}}, {{3, 3, 0}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 1);
  CHECK(c.ef == 1);
  CHECK_SELF_MATCHES(a, b, c);
}

// =============================================================================
// Fan orient3d zero -> crossing_edges_vs_face classification
// =============================================================================

TEST_CASE("Edge through face edge (EE)", "[exact][ee]") {
  auto a = make_mesh<4>(
      {{{0, -2, 0}}, {{0, 2, 0}}, {{4, 2, 0}}, {{4, -2, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{0, 0, 3}}, {{0, 0, -3}}, {{-3, 0, 0}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 1);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Edge through vertex - no duplicate VE", "[exact][ve]") {
  auto a = make_mesh<4>(
      {{{-2, 0, 0}}, {{2, 0, 0}}, {{2, 4, 0}}, {{-2, 4, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{0, 0, 0}}, {{0, 2, 3}}, {{0, -2, 3}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 1);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

// =============================================================================
// Mixed non-coplanar type combinations
// =============================================================================

TEST_CASE("VF + EF (vertex in face, edge crosses face)", "[exact][mixed]") {
  auto a = make_mesh<4>(
      {{{-3, -3, 0}}, {{3, -3, 0}}, {{3, 3, 0}}, {{-3, 3, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{1, 1, 0}}, {{0, -1, 3}}, {{0, -1, -3}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 1);
  CHECK(c.ef == 1);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("VV + EF (shared vertex, edge crosses face)", "[exact][mixed]") {
  auto a = make_mesh<4>(
      {{{-3, -3, 0}}, {{3, -3, 0}}, {{3, 3, 0}}, {{-3, 3, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{-3, -3, 0}}, {{1, 0, 3}}, {{1, 0, -3}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 1);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 1);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("VF + EE (vertex in face, edge hits edge)", "[exact][mixed]") {
  auto a = make_mesh<4>(
      {{{-3, -3, 0}}, {{3, -3, 0}}, {{3, 3, 0}}, {{-3, 3, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{1, 1, 0}}, {{3, 0, 3}}, {{3, 0, -3}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 1);
  CHECK(c.vf == 1);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("VV + EE (shared vertex, edge hits edge)", "[exact][mixed]") {
  auto a = make_mesh<4>(
      {{{-3, -3, 0}}, {{3, -3, 0}}, {{3, 3, 0}}, {{-3, 3, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{-3, -3, 0}}, {{0, 3, 3}}, {{0, 3, -3}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 1);
  CHECK(c.ve == 0);
  CHECK(c.ee == 1);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("VF + VE (vertex in face, edge through vertex)", "[exact][mixed]") {
  auto a = make_mesh<4>(
      {{{-3, -3, 0}}, {{3, -3, 0}}, {{3, 3, 0}}, {{-3, 3, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{1, 1, 0}}, {{-3, -3, 3}}, {{-3, -3, -3}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 1);
  CHECK(c.ee == 0);
  CHECK(c.vf == 1);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("VV + VE (shared vertex, edge through vertex)", "[exact][mixed]") {
  auto a = make_mesh<4>(
      {{{-3, -3, 0}}, {{3, -3, 0}}, {{3, 3, 0}}, {{-3, 3, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{-3, -3, 0}}, {{3, 3, 3}}, {{3, 3, -3}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 1);
  CHECK(c.ve == 1);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

// =============================================================================
// N-mesh tests (range overload + dedup)
// =============================================================================

TEST_CASE("3 identical quads - 4 VV only", "[exact][nmesh]") {
  auto q = make_mesh<4>(
      {{{-1, -1, 0}}, {{1, -1, 0}}, {{1, 1, 0}}, {{-1, 1, 0}}},
      {{{0, 1, 2, 3}}});

  tf::aabb_tree<int, float, 3> ta(q.polygons(), tf::config_tree(4, 4));
  auto fm = tf::make_face_membership(q.polygons());
  auto mel = tf::make_manifold_edge_link(q.polygons());
  auto tagged = q.polygons() | tf::tag(ta) | tf::tag(fm) | tf::tag(mel);

  decltype(tagged) forms[] = {tagged, tagged, tagged};
  auto c = count_primitives_n(forms, 3);
  CHECK(c.vv == 4);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES_N(forms, 3, c);
}

TEST_CASE("3 mesh: 2 identical + 1 inside - 4 VV + 4 VF", "[exact][nmesh]") {
  auto big = make_mesh<4>(
      {{{-3, -3, 0}}, {{3, -3, 0}}, {{3, 3, 0}}, {{-3, 3, 0}}},
      {{{0, 1, 2, 3}}});
  auto small_mesh = make_mesh<4>(
      {{{-1, -1, 0}}, {{1, -1, 0}}, {{1, 1, 0}}, {{-1, 1, 0}}},
      {{{0, 1, 2, 3}}});

  tf::aabb_tree<int, float, 3> t0(big.polygons(), tf::config_tree(4, 4));
  tf::aabb_tree<int, float, 3> t1(big.polygons(), tf::config_tree(4, 4));
  tf::aabb_tree<int, float, 3> t2(small_mesh.polygons(),
                                   tf::config_tree(4, 4));
  auto fm0 = tf::make_face_membership(big.polygons());
  auto fm1 = tf::make_face_membership(big.polygons());
  auto fm2 = tf::make_face_membership(small_mesh.polygons());
  auto mel0 = tf::make_manifold_edge_link(big.polygons());
  auto mel1 = tf::make_manifold_edge_link(big.polygons());
  auto mel2 = tf::make_manifold_edge_link(small_mesh.polygons());
  auto tag0 = big.polygons() | tf::tag(t0) | tf::tag(fm0) | tf::tag(mel0);
  auto tag1 = big.polygons() | tf::tag(t1) | tf::tag(fm1) | tf::tag(mel1);
  auto tag2 =
      small_mesh.polygons() | tf::tag(t2) | tf::tag(fm2) | tf::tag(mel2);

  decltype(tag0) forms[] = {tag0, tag1, tag2};
  auto c = count_primitives_n(forms, 3);
  CHECK(c.vv == 4);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 4);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES_N(forms, 3, c);
}

TEST_CASE("3 mesh: shared vertex dedup - 4 VV", "[exact][nmesh]") {
  auto q = make_mesh<4>(
      {{{-1, -1, 0}}, {{1, -1, 0}}, {{1, 1, 0}}, {{-1, 1, 0}}},
      {{{0, 1, 2, 3}}});
  auto corner = make_mesh<4>(
      {{{-1, -1, 0}}, {{-1, -3, 0}}, {{-3, -3, 0}}, {{-3, -1, 0}}},
      {{{0, 1, 2, 3}}});

  tf::aabb_tree<int, float, 3> t0(q.polygons(), tf::config_tree(4, 4));
  tf::aabb_tree<int, float, 3> t1(q.polygons(), tf::config_tree(4, 4));
  tf::aabb_tree<int, float, 3> t2(corner.polygons(), tf::config_tree(4, 4));
  auto fm0 = tf::make_face_membership(q.polygons());
  auto fm1 = tf::make_face_membership(q.polygons());
  auto fm2 = tf::make_face_membership(corner.polygons());
  auto mel0 = tf::make_manifold_edge_link(q.polygons());
  auto mel1 = tf::make_manifold_edge_link(q.polygons());
  auto mel2 = tf::make_manifold_edge_link(corner.polygons());
  auto tag0 = q.polygons() | tf::tag(t0) | tf::tag(fm0) | tf::tag(mel0);
  auto tag1 = q.polygons() | tf::tag(t1) | tf::tag(fm1) | tf::tag(mel1);
  auto tag2 =
      corner.polygons() | tf::tag(t2) | tf::tag(fm2) | tf::tag(mel2);

  decltype(tag0) forms[] = {tag0, tag1, tag2};
  auto c = count_primitives_n(forms, 3);
  CHECK(c.vv == 4);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES_N(forms, 3, c);
}

TEST_CASE("3 mesh: 2 identical + 1 crossing - 4 VV + 2 EF", "[exact][nmesh]") {
  auto q = make_mesh<4>(
      {{{-3, -3, 0}}, {{3, -3, 0}}, {{3, 3, 0}}, {{-3, 3, 0}}},
      {{{0, 1, 2, 3}}});
  auto cross = make_mesh<4>(
      {{{1, 0, 3}}, {{1, 0, -3}}, {{5, 0, -3}}, {{5, 0, 3}}},
      {{{0, 1, 2, 3}}});

  tf::aabb_tree<int, float, 3> t0(q.polygons(), tf::config_tree(4, 4));
  tf::aabb_tree<int, float, 3> t1(q.polygons(), tf::config_tree(4, 4));
  tf::aabb_tree<int, float, 3> t2(cross.polygons(), tf::config_tree(4, 4));
  auto fm0 = tf::make_face_membership(q.polygons());
  auto fm1 = tf::make_face_membership(q.polygons());
  auto fm2 = tf::make_face_membership(cross.polygons());
  auto mel0 = tf::make_manifold_edge_link(q.polygons());
  auto mel1 = tf::make_manifold_edge_link(q.polygons());
  auto mel2 = tf::make_manifold_edge_link(cross.polygons());
  auto tag0 = q.polygons() | tf::tag(t0) | tf::tag(fm0) | tf::tag(mel0);
  auto tag1 = q.polygons() | tf::tag(t1) | tf::tag(fm1) | tf::tag(mel1);
  auto tag2 =
      cross.polygons() | tf::tag(t2) | tf::tag(fm2) | tf::tag(mel2);

  decltype(tag0) forms[] = {tag0, tag1, tag2};
  auto c = count_primitives_n(forms, 3);
  CHECK(c.vv == 4);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 2);
  // Self: coincident intersection points across the two identical quads
  // share one ID after geometric dedup → matches between (4 VV, 2 EF).
  auto s = count_self_primitives_n(forms, 3);
  CHECK(s.vv == 4);
  CHECK(s.ve == 0);
  CHECK(s.ee == 0);
  CHECK(s.vf == 0);
  CHECK(s.ef == 2);
}

// =============================================================================
// Fan-based crossing classification tests
// =============================================================================

TEST_CASE("Fan: edge on diagonal -> EF", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 0, 0}}, {{4, 4, 0}}, {{0, 4, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{2, 2, 3}}, {{2, 2, -3}}, {{8, 8, 0}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 1);
  CHECK(c.ef == 1);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Fan: edge on real edge -> EE", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 0, 0}}, {{4, 4, 0}}, {{0, 4, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{4, 2, 3}}, {{4, 2, -3}}, {{8, 2, 0}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 1);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Fan: edge through vertex -> VE", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 0, 0}}, {{4, 4, 0}}, {{0, 4, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{4, 4, 3}}, {{4, 4, -3}}, {{8, 8, 0}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 1);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Fan: edge through fan apex -> VE", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 0, 0}}, {{4, 4, 0}}, {{0, 4, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{0, 0, 3}}, {{0, 0, -3}}, {{-4, 0, 0}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 1);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Fan: edge on first real edge -> EE", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 0, 0}}, {{4, 4, 0}}, {{0, 4, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{2, 0, 3}}, {{2, 0, -3}}, {{2, -4, 0}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 1);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Fan: edge on last real edge -> EE", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 0, 0}}, {{4, 4, 0}}, {{0, 4, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{0, 2, 3}}, {{0, 2, -3}}, {{-4, 2, 0}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 1);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Two quads: both EE (edges on edges)", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{-3, -3, 0}}, {{3, -3, 0}}, {{3, 3, 0}}, {{-3, 3, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{0, -3, -3}}, {{0, 3, -3}}, {{0, 3, 3}}, {{0, -3, 3}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 2);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Two quads: both EF (interior crossing)", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{-5, -5, 0}}, {{5, -5, 0}}, {{5, 5, 0}}, {{-5, 5, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{-2, 0, -2}}, {{2, 0, -2}}, {{2, 0, 2}}, {{-2, 0, 2}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 2);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Quad+tri: EE on edge + EF interior", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{-3, -3, 0}}, {{3, -3, 0}}, {{3, 3, 0}}, {{-3, 3, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{0, 3, 3}}, {{0, 3, -3}}, {{0, 0, -3}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 1);
  CHECK(c.vf == 0);
  CHECK(c.ef == 1);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Quad+tri: VE at vertex + EF interior", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{-3, -3, 0}}, {{3, -3, 0}}, {{3, 3, 0}}, {{-3, 3, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<3>(
      {{{3, 3, 3}}, {{3, 3, -3}}, {{0, 0, -3}}}, {{{0, 1, 2}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 1);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 1);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Fan: quad over diagonal -> 2 VE", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 0, 0}}, {{4, 4, 0}}, {{0, 4, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{0, 0, 2}}, {{4, 4, 2}}, {{4, 4, -2}}, {{0, 0, -2}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 2);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Fan: quad over diagonal -> 2 VV", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 0, 0}}, {{4, 4, 0}}, {{0, 4, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 4, 0}}, {{4, 4, 2}}, {{0, 0, 2}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 2);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Vertical quad on face -> 2 VF", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{-4, -4, 0}}, {{4, -4, 0}}, {{4, 4, 0}}, {{-4, 4, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{1, 0, 0}}, {{3, 0, 0}}, {{3, 0, 2}}, {{1, 0, 2}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 2);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Vertical quad on edges -> 2 VE", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 0, 0}}, {{4, 4, 0}}, {{0, 4, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{0, 2, 0}}, {{4, 2, 0}}, {{4, 2, 2}}, {{0, 2, 2}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 2);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Vertical quad extends over -> 2 EE", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 0, 0}}, {{4, 4, 0}}, {{0, 4, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{-1, 2, 0}}, {{5, 2, 0}}, {{5, 2, 2}}, {{-1, 2, 2}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 2);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Vertical quad on edge - no crossing", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 0, 0}}, {{4, 4, 0}}, {{0, 4, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{0, 0, 0}}, {{4, 0, 0}}, {{4, 0, 2}}, {{0, 0, 2}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 2);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
  CHECK_SELF_MATCHES(a, b, c);
}

TEST_CASE("Perp quads: 1 EE + 1 EF", "[exact][fan]") {
  auto a = make_mesh<4>(
      {{{-3, -3, 0}}, {{3, -3, 0}}, {{3, 3, 0}}, {{-3, 3, 0}}},
      {{{0, 1, 2, 3}}});
  auto b = make_mesh<4>(
      {{{-1, 0, -3}}, {{3, 0, -3}}, {{3, 0, 3}}, {{-1, 0, 3}}},
      {{{0, 1, 2, 3}}});
  auto c = count_primitives(a, b);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 1);
  CHECK(c.vf == 0);
  CHECK(c.ef == 1);
  CHECK_SELF_MATCHES(a, b, c);
}

// =============================================================================
// Self-intersection: shared edge tests
// =============================================================================

template <typename Mesh>
auto count_within(const Mesh &mesh) -> type_counts {
  tf::aabb_tree<int, float, 3> tree(mesh.polygons(), tf::config_tree(4, 4));
  auto fm = tf::make_face_membership(mesh.polygons());
  auto mel = tf::make_manifold_edge_link(mesh.polygons());
  auto tagged = mesh.polygons() | tf::tag(tree) | tf::tag(fm) | tf::tag(mel);
  tf::intersections_within_polygons<Index, float> iwp;
  iwp.build(tagged, tf::intersect_mode::primitives);
  std::set<int> s_vv, s_ve, s_ee, s_vf, s_ef;
  for (auto subrange : iwp.intersections()) {
    for (auto &r : subrange) {
      int id = r.id;
      auto a = r.target.label, b = r.target_other.label;
      using T = tf::topo_type;
      if (a == T::vertex && b == T::vertex) s_vv.insert(id);
      else if ((a == T::vertex && b == T::edge) || (a == T::edge && b == T::vertex)) s_ve.insert(id);
      else if (a == T::edge && b == T::edge) s_ee.insert(id);
      else if ((a == T::vertex && b == T::face) || (a == T::face && b == T::vertex)) s_vf.insert(id);
      else if ((a == T::edge && b == T::face) || (a == T::face && b == T::edge)) s_ef.insert(id);
    }
  }
  return {int(s_vv.size()), int(s_ve.size()), int(s_ee.size()),
          int(s_vf.size()), int(s_ef.size())};
}

TEST_CASE("Self: shared edge, non-coplanar, no intersection", "[exact][self]") {
  auto mesh = make_mesh<3>(
      {{{0,0,0}}, {{1,0,0}}, {{0.5f,1,0.5f}}, {{0.5f,-1,-0.5f}}},
      {{{0,1,2}}, {{0,1,3}}});
  auto c = count_within(mesh);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
}

TEST_CASE("Self: shared edge, coplanar same coords - VV", "[exact][self]") {
  auto mesh = make_mesh<3>(
      {{{0,0,0}}, {{1,0,0}}, {{0.5f,1,0}}, {{0.5f,1,0}}},
      {{{0,1,2}}, {{0,1,3}}});
  auto c = count_within(mesh);
  CHECK(c.vv == 1);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
}

TEST_CASE("Self: shared edge, coplanar inside face - VF", "[exact][self]") {
  auto mesh = make_mesh<3>(
      {{{0,0,0}}, {{1,0,0}}, {{0.5f,1,0}}, {{0.3f,0.3f,0}}},
      {{{0,1,2}}, {{0,1,3}}});
  auto c = count_within(mesh);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 0);
  CHECK(c.vf == 1);
  CHECK(c.ef == 0);
}

TEST_CASE("Self: shared edge, coplanar on edge - VE", "[exact][self]") {
  auto mesh = make_mesh<3>(
      {{{0,0,0}}, {{1,0,0}}, {{0.5f,1,0}}, {{0.25f,0.5f,0}}},
      {{{0,1,2}}, {{0,1,3}}});
  auto c = count_within(mesh);
  CHECK(c.vv == 0);
  CHECK(c.ve == 1);
  CHECK(c.ee == 0);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
}

TEST_CASE("Self: shared edge, coplanar EE crossing", "[exact][self]") {
  auto mesh = make_mesh<3>(
      {{{0,0,0}}, {{1,0,0}}, {{0.5f,1,0}}, {{1.5f,0.5f,0}}},
      {{{0,1,2}}, {{0,1,3}}});
  auto c = count_within(mesh);
  CHECK(c.vv == 0);
  CHECK(c.ve == 0);
  CHECK(c.ee == 1);
  CHECK(c.vf == 0);
  CHECK(c.ef == 0);
}

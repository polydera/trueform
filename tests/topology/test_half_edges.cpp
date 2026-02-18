/**
 * @file test_half_edges.cpp
 * @brief Tests for half-edge data structure: build, collapse, link condition
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "topology_generators.hpp"
#include "type_traits.hpp"

namespace {

/// Find edge handle for edge {va, vb} in a half-edge structure.
template <typename Index>
auto find_edge(const tf::half_edges<Index> &he, Index va, Index vb)
    -> tf::edge_handle<Index> {
  for (auto eh : he.edge_handles()) {
    if (!eh.is_valid())
      continue;
    auto heh0 = he.half_edge_handle(tf::unsafe, eh, false);
    auto v0 = he.start_vertex_handle(tf::unsafe, heh0).id();
    auto v1 = he.end_vertex_handle(tf::unsafe, heh0).id();
    if ((v0 == va && v1 == vb) || (v0 == vb && v1 == va))
      return eh;
  }
  return tf::edge_handle<Index>::invalid();
}

/// Find half-edge handle from va to vb (va survives on collapse).
template <typename Index>
auto find_half_edge(const tf::half_edges<Index> &he, Index va, Index vb)
    -> tf::half_edge_handle<Index> {
  auto eh = find_edge(he, va, vb);
  if (!eh.is_valid())
    return tf::half_edge_handle<Index>::invalid();
  auto heh0 = he.half_edge_handle(tf::unsafe, eh, false);
  auto v0 = he.start_vertex_handle(tf::unsafe, heh0).id();
  return (v0 == va) ? heh0 : he.opposite(tf::unsafe, heh0);
}

/// Count valid/invalid edges.
template <typename Index>
auto count_edges(const tf::half_edges<Index> &he)
    -> std::pair<Index, Index> {
  Index valid = 0, invalid = 0;
  for (auto eh : he.edge_handles()) {
    if (eh.is_valid())
      ++valid;
    else
      ++invalid;
  }
  return {valid, invalid};
}

/// Count valid/invalid face handles.
template <typename Index>
auto count_faces(const tf::half_edges<Index> &he)
    -> std::pair<Index, Index> {
  Index valid = 0, invalid = 0;
  for (auto fhe : he.face_half_edge_handles()) {
    if (fhe.is_valid())
      ++valid;
    else
      ++invalid;
  }
  return {valid, invalid};
}

/// Collect ring vertices around a vertex by rotation.
template <typename Index>
auto collect_ring(const tf::half_edges<Index> &he, Index vertex)
    -> tf::hash_set<Index> {
  tf::hash_set<Index> ring;
  auto start = he.vertex_half_edge_handles()[vertex];
  if (!start.is_valid())
    return ring;
  auto cur = start;
  Index count = 0;
  do {
    ring.insert(he.end_vertex_handle(tf::unsafe, cur).id());
    ++count;
    cur = he.rotated(cur);
    if (!cur.is_valid())
      break;
  } while (cur != start && count < 100);
  return ring;
}

/// Check that all valid faces have consistent 3-half-edge loops.
template <typename Index>
auto check_face_loops(const tf::half_edges<Index> &he) -> bool {
  auto &&face_hes = he.face_half_edge_handles();
  for (Index f = 0; f < Index(face_hes.size()); ++f) {
    if (!face_hes[f].is_valid())
      continue;
    auto cur = face_hes[f];
    Index count = 0;
    do {
      if (he.face_handle(tf::unsafe, cur).id() != f)
        return false;
      cur = he.next(tf::unsafe, cur);
      ++count;
    } while (cur != face_hes[f] && count < 10);
    if (count != 3)
      return false;
  }
  return true;
}

/// Check that no valid half-edge references a given vertex.
template <typename Index>
auto no_half_edge_references_vertex(const tf::half_edges<Index> &he,
                                    Index v) -> bool {
  for (auto heh : he.half_edge_handles()) {
    if (!heh.is_valid())
      continue;
    if (he.start_vertex_handle(tf::unsafe, heh).id() == v)
      return false;
  }
  return true;
}

} // namespace

// =============================================================================
// Interior edge collapse
// =============================================================================

TEMPLATE_TEST_CASE("half_edges_interior_collapse", "[topology][half_edges]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using Index = typename TestType::index_type;
  using Real = typename TestType::real_type;

  auto mesh = tf::test::create_diamond_mesh_3d<Index, Real>();
  tf::half_edges<Index> he(mesh.polygons());

  auto target_eh = find_edge(he, Index(0), Index(1));
  REQUIRE(target_eh.is_valid());
  REQUIRE(he.is_manifold(target_eh));
  REQUIRE_FALSE(he.is_boundary(target_eh));

  tf::small_vector<Index, 16> ring;
  REQUIRE(he.is_collapse_ok(target_eh, ring));

  auto target_heh = find_half_edge(he, Index(0), Index(1));
  he.collapse(target_heh);

  auto [valid_edges, removed_edges] = count_edges(he);
  REQUIRE(removed_edges == 3);

  auto [valid_faces, invalid_faces] = count_faces(he);
  REQUIRE(invalid_faces == 2);
  REQUIRE(valid_faces == 4);

  REQUIRE_FALSE(he.vertex_half_edge_handles()[1].is_valid());
  auto v0_he = he.vertex_half_edge_handles()[0];
  REQUIRE(v0_he.is_valid());
  REQUIRE(he.start_vertex_handle(tf::unsafe, v0_he).id() == 0);

  REQUIRE(check_face_loops(he));

  auto ring_verts = collect_ring(he, Index(0));
  REQUIRE(ring_verts.count(2));
  REQUIRE(ring_verts.count(3));
  REQUIRE(ring_verts.count(4));
  REQUIRE(ring_verts.count(5));
  REQUIRE(ring_verts.count(6));
  REQUIRE_FALSE(ring_verts.count(1));

  REQUIRE(no_half_edge_references_vertex(he, Index(1)));
}

// =============================================================================
// Link condition failure
// =============================================================================

TEMPLATE_TEST_CASE("half_edges_link_condition_failure",
                   "[topology][half_edges]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using Index = typename TestType::index_type;
  using Real = typename TestType::real_type;

  auto mesh = tf::test::create_link_condition_fail_mesh_3d<Index, Real>();
  tf::half_edges<Index> he(mesh.polygons());

  auto target_eh = find_edge(he, Index(0), Index(1));
  REQUIRE(target_eh.is_valid());

  tf::small_vector<Index, 16> ring;
  REQUIRE_FALSE(he.is_collapse_ok(target_eh, ring));
}

// =============================================================================
// Non-manifold edge in 1-ring
// =============================================================================

TEMPLATE_TEST_CASE("half_edges_non_manifold_ring",
                   "[topology][half_edges]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using Index = typename TestType::index_type;
  using Real = typename TestType::real_type;

  auto mesh = tf::test::create_non_manifold_ring_mesh_3d<Index, Real>();
  tf::half_edges<Index> he(mesh.polygons());

  auto target_eh = find_edge(he, Index(0), Index(1));
  REQUIRE(target_eh.is_valid());
  REQUIRE(he.is_manifold(target_eh));

  tf::small_vector<Index, 16> ring;
  REQUIRE_FALSE(he.is_collapse_ok(target_eh, ring));
}

// =============================================================================
// Boundary edge collapse
// =============================================================================

TEMPLATE_TEST_CASE("half_edges_boundary_collapse", "[topology][half_edges]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using Index = typename TestType::index_type;
  using Real = typename TestType::real_type;

  auto mesh = tf::test::create_boundary_collapse_mesh_3d<Index, Real>();
  tf::half_edges<Index> he(mesh.polygons());

  auto target_eh = find_edge(he, Index(0), Index(1));
  REQUIRE(target_eh.is_valid());
  REQUIRE(he.is_boundary(target_eh));

  tf::small_vector<Index, 16> ring;
  REQUIRE(he.is_collapse_ok(target_eh, ring));

  auto target_heh = find_half_edge(he, Index(0), Index(1));
  he.collapse(target_heh);

  auto [valid_edges, removed_edges] = count_edges(he);
  REQUIRE(removed_edges == 2);

  auto [valid_faces, invalid_faces] = count_faces(he);
  REQUIRE(invalid_faces == 1);
  REQUIRE(valid_faces == 2);

  REQUIRE_FALSE(he.vertex_half_edge_handles()[1].is_valid());
  auto v0_he = he.vertex_half_edge_handles()[0];
  REQUIRE(v0_he.is_valid());
  REQUIRE(he.start_vertex_handle(tf::unsafe, v0_he).id() == 0);

  REQUIRE(check_face_loops(he));

  auto ring_verts = collect_ring(he, Index(0));
  REQUIRE(ring_verts.count(2));
  REQUIRE(ring_verts.count(3));
  REQUIRE(ring_verts.count(4));
  REQUIRE_FALSE(ring_verts.count(1));

  REQUIRE(no_half_edge_references_vertex(he, Index(1)));
}

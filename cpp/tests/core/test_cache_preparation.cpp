/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#include "carriers.hpp"

#include "trueform/cpp/core/build_face_link.hpp"
#include "trueform/cpp/core/build_face_membership.hpp"
#include "trueform/cpp/core/build_face_normals.hpp"
#include "trueform/cpp/core/build_half_edges.hpp"
#include "trueform/cpp/core/build_manifold_edge_link.hpp"
#include "trueform/cpp/core/build_point_normals.hpp"
#include "trueform/cpp/core/build_tree.hpp"
#include "trueform/cpp/core/build_vertex_link.hpp"
#include "trueform/cpp/core/build_winding_moments.hpp"
#include "trueform/cpp/core/detail/fill_guard.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using prep_index = tf::cpp::default_index_t;

template <typename Real>
using prep_owned = tf::cpp::test::owned_mesh<prep_index, Real>;

template <typename Real>
using prep_mixed_owned =
    tf::cpp::test::owned_mesh<prep_index, Real, 3, tf::dynamic_size>;

template <typename Real>
auto triangle_mesh(Real offset = 0) -> prep_owned<Real> {
  return {tf::cpp::test::polygons_of<prep_index, Real>(
      {0, 1, 2}, {offset + 0, 0, 0, offset + 1, 0, 0, offset + 0, 1, 0})};
}

template <typename Real> auto mixed_mesh() -> prep_mixed_owned<Real> {
  return {tf::cpp::test::polygons_of<prep_index, Real>(
      {0, 3, 7}, {0, 1, 2, 1, 3, 4, 2},
      {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0, 3, 3, 0})};
}

/// The caller's own storage, restated: a test writes through it and then says
/// what moved, which is the protocol the cache carries.
template <typename Owned, typename Real>
auto restate_points(Owned &owned, std::initializer_list<Real> coordinates)
    -> void {
  auto &storage = owned.polygons.points_buffer().data_buffer();
  storage.allocate(coordinates.size());
  std::copy(coordinates.begin(), coordinates.end(), storage.begin());
  owned.cache.points_changed();
}

template <typename Owned>
auto restate_mixed_faces(Owned &owned,
                         std::initializer_list<prep_index> offsets,
                         std::initializer_list<prep_index> indices) -> void {
  auto &spans = owned.polygons.faces_buffer().offsets_buffer();
  spans.allocate(offsets.size());
  std::copy(offsets.begin(), offsets.end(), spans.begin());
  auto &corners = owned.polygons.faces_buffer().data_buffer();
  corners.allocate(indices.size());
  std::copy(indices.begin(), indices.end(), corners.begin());
  owned.cache.faces_changed();
}

/// A mesh whose caches take long enough to build that a naive caller would
/// have two fills in flight; the contract is what keeps them apart.
auto cache_race_grid_mesh(int side) -> prep_owned<float> {
  const auto rows = side + 1;
  prep_owned<float> owned;
  auto &points = owned.polygons.points_buffer().data_buffer();
  points.allocate(static_cast<std::size_t>(rows) * rows * 3);
  auto *coordinate = points.begin();
  for (int y = 0; y != rows; ++y)
    for (int x = 0; x != rows; ++x) {
      *coordinate++ = static_cast<float>(x);
      *coordinate++ = static_cast<float>(y);
      *coordinate++ = static_cast<float>((x * 7 + y * 13) % 5);
    }
  auto &faces = owned.polygons.faces_buffer().data_buffer();
  faces.allocate(static_cast<std::size_t>(side) * side * 6);
  auto *corner = faces.begin();
  for (int y = 0; y != side; ++y)
    for (int x = 0; x != side; ++x) {
      const auto first = y * rows + x;
      *corner++ = first;
      *corner++ = first + 1;
      *corner++ = first + rows;
      *corner++ = first + 1;
      *corner++ = first + rows + 1;
      *corner++ = first + rows;
    }
  return owned;
}

template <typename Blocks>
auto sorted_block(const Blocks &blocks, std::size_t block_id)
    -> std::vector<prep_index> {
  std::vector<prep_index> output;
  for (const auto value : blocks[block_id])
    output.push_back(value);
  std::sort(output.begin(), output.end());
  return output;
}

template <typename HalfEdges>
auto check_face_cycle(
    const HalfEdges &half_edges, prep_index face_id,
    std::initializer_list<std::pair<prep_index, prep_index>> expected_edges)
    -> void {
  const auto start = half_edges.face_half_edge_handles()[face_id];
  REQUIRE(start.is_valid());
  auto current = start;
  std::vector<std::pair<prep_index, prep_index>> actual_edges;
  for (std::size_t i = 0; i < expected_edges.size(); ++i) {
    REQUIRE(current.is_valid());
    CHECK(half_edges.face_handle(current).id() == face_id);
    actual_edges.emplace_back(half_edges.start_vertex_handle(current).id(),
                              half_edges.end_vertex_handle(current).id());
    const auto next = half_edges.next(current);
    REQUIRE(next.is_valid());
    CHECK(half_edges.previous(next) == current);
    CHECK(half_edges.end_vertex_handle(current) ==
          half_edges.start_vertex_handle(next));
    if (i + 1 < expected_edges.size())
      CHECK(next != start);
    else
      CHECK(next == start);
    current = next;
  }
  std::sort(actual_edges.begin(), actual_edges.end());
  auto expected = std::vector<std::pair<prep_index, prep_index>>(
      expected_edges.begin(), expected_edges.end());
  std::sort(expected.begin(), expected.end());
  CHECK(actual_edges == expected);
}

} // namespace

TEMPLATE_TEST_CASE("the build verbs fill a mesh's and a cloud's caches",
                   "[cpp][core][cache]", float, double) {
  const auto owned = triangle_mesh<TestType>();
  const auto mesh = owned.mesh();

  tf::cpp::build_tree(mesh);
  tf::cpp::build_winding_moments(mesh);
  tf::cpp::build_half_edges(mesh);
  tf::cpp::build_face_membership(mesh);
  tf::cpp::build_manifold_edge_link(mesh);
  tf::cpp::build_face_link(mesh);
  tf::cpp::build_vertex_link(mesh);
  tf::cpp::build_face_normals(mesh);
  tf::cpp::build_point_normals(mesh);

  CHECK(owned.cache.is_tree_fresh(mesh.geometry()));
  CHECK(owned.cache.is_winding_moments_fresh(mesh.geometry()));
  CHECK(owned.cache.is_half_edges_fresh(mesh.geometry()));
  CHECK(owned.cache.is_face_membership_fresh(mesh.geometry()));
  CHECK(owned.cache.is_manifold_edge_link_fresh(mesh.geometry()));
  CHECK(owned.cache.is_face_link_fresh(mesh.geometry()));
  CHECK(owned.cache.is_vertex_link_fresh(mesh.geometry()));
  CHECK(owned.cache.is_face_normals_fresh(mesh.geometry()));
  CHECK(owned.cache.is_point_normals_fresh(mesh.geometry()));

  const tf::cpp::test::owned_point_cloud<TestType> cloud{
      tf::cpp::test::points_of<TestType>({0, 0, 0, 1, 0, 0, 0, 1, 0})};
  const auto points = cloud.point_cloud();
  tf::cpp::build_tree(points);
  CHECK(cloud.cache.is_tree_fresh(points.geometry()));
}

TEMPLATE_TEST_CASE("a mixed mesh exposes exact mesh-built topology caches",
                   "[cpp][core][cache][carrier-integration]", float, double) {
  const auto owned = mixed_mesh<TestType>();
  const auto mesh = owned.mesh();

  tf::cpp::build_tree(mesh);
  tf::cpp::build_face_membership(mesh);
  tf::cpp::build_face_link(mesh);
  tf::cpp::build_vertex_link(mesh);
  const auto &half_edges = mesh.half_edges();

  REQUIRE(owned.cache.is_tree_fresh(mesh.geometry()));
  REQUIRE(owned.cache.is_face_membership_fresh(mesh.geometry()));
  REQUIRE(owned.cache.is_half_edges_fresh(mesh.geometry()));
  REQUIRE(owned.cache.is_face_link_fresh(mesh.geometry()));
  REQUIRE(owned.cache.is_vertex_link_fresh(mesh.geometry()));

  const auto membership = mesh.face_membership();
  REQUIRE(membership.size() == 6);
  CHECK(sorted_block(membership, 0) == std::vector<prep_index>{0});
  CHECK(sorted_block(membership, 1) == std::vector<prep_index>{0, 1});
  CHECK(sorted_block(membership, 2) == std::vector<prep_index>{0, 1});
  CHECK(sorted_block(membership, 3) == std::vector<prep_index>{1});
  CHECK(sorted_block(membership, 4) == std::vector<prep_index>{1});
  CHECK(membership[5].size() == 0);

  const auto primitive_bounds = mesh.tree().primitive_aabbs();
  REQUIRE(primitive_bounds.size() == 2);
  CHECK(primitive_bounds[0].min[0] == static_cast<TestType>(0));
  CHECK(primitive_bounds[0].max[0] == static_cast<TestType>(1));
  CHECK(primitive_bounds[1].min[0] == static_cast<TestType>(0));
  CHECK(primitive_bounds[1].max[0] == static_cast<TestType>(2));
  CHECK(primitive_bounds[1].max[1] == static_cast<TestType>(1));

  CHECK(half_edges.number_of_faces() == 2);
  CHECK(half_edges.number_of_vertices() == 6);
  CHECK(half_edges.face_half_edge_handles().size() == 2);
  CHECK(half_edges.vertex_half_edge_handles().size() == 6);
  CHECK(half_edges.half_edges_buffer().size() == 12);
  CHECK_FALSE(half_edges.vertex_half_edge_handles()[5].is_valid());
  check_face_cycle(half_edges, 0, {{0, 1}, {1, 2}, {2, 0}});
  check_face_cycle(half_edges, 1, {{1, 3}, {3, 4}, {4, 2}, {2, 1}});

  const auto face_link = mesh.face_link();
  REQUIRE(face_link.size() == 2);
  CHECK(sorted_block(face_link, 0) == std::vector<prep_index>{1});
  CHECK(sorted_block(face_link, 1) == std::vector<prep_index>{0});

  const auto vertex_link = mesh.vertex_link();
  REQUIRE(vertex_link.size() == 6);
  CHECK(sorted_block(vertex_link, 0) == std::vector<prep_index>{1, 2});
  CHECK(sorted_block(vertex_link, 1) == std::vector<prep_index>{0, 2, 3});
  CHECK(sorted_block(vertex_link, 2) == std::vector<prep_index>{0, 1, 4});
  CHECK(sorted_block(vertex_link, 3) == std::vector<prep_index>{1, 4});
  CHECK(sorted_block(vertex_link, 4) == std::vector<prep_index>{2, 3});
  CHECK(vertex_link[5].size() == 0);

  CHECK(owned.cache.face_link_build_count() == 1);
  CHECK(owned.cache.vertex_link_build_count() == 1);
  CHECK(owned.cache.tree_build_count() == 1);
  CHECK(owned.cache.face_membership_build_count() == 1);

  const auto form = mesh.form();
  CHECK(std::array<TestType, 4>{static_cast<TestType>(form.size()),
                                static_cast<TestType>(form[0].size()),
                                static_cast<TestType>(form[1].size()),
                                form.tree().bv().max[0]} ==
        std::array<TestType, 4>{2, 3, 4, 2});
  CHECK(owned.cache.tree_build_count() == 1);
  // a peer per face side is what a face has at either arity, and the mesh is
  // what reads the one flat array back at the arity it is stored in
  const auto peers = mesh.manifold_edge_link();
  REQUIRE(peers.size() == 2);
  CHECK(peers[0].size() == 3);
  CHECK(peers[1].size() == 4);

  // every structure exists at either arity, so a caller may state its own —
  // and a malformed one is refused where it is stated
  CHECK_THROWS_AS(owned.cache.set_half_edges({}, mesh.geometry()),
                  std::invalid_argument);
  CHECK_THROWS_AS(owned.cache.set_manifold_edge_link({}, mesh.geometry()),
                  std::invalid_argument);
  CHECK_THROWS_AS(owned.cache.set_face_link({}, mesh.geometry()),
                  std::invalid_argument);
  CHECK_THROWS_AS(owned.cache.set_vertex_link({}, mesh.geometry()),
                  std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "a stated change rebuilds only the structures that depended on it",
    "[cpp][core][cache][cache-authority]", float, double) {
  auto owned = mixed_mesh<TestType>();

  {
    const auto mesh = owned.mesh();
    static_cast<void>(mesh.half_edges());
    static_cast<void>(mesh.face_link());
    static_cast<void>(mesh.vertex_link());
    static_cast<void>(mesh.half_edges());
    static_cast<void>(mesh.face_link());
    static_cast<void>(mesh.vertex_link());
  }
  CHECK(owned.cache.half_edges_build_count() == 1);
  CHECK(owned.cache.face_link_build_count() == 1);
  CHECK(owned.cache.vertex_link_build_count() == 1);

  // THE TWO STAMPS: a point that MOVED is not a point that was ADDED, and a
  // structure that only counts points is fresh across the first
  owned.polygons.points_buffer().data_buffer()[0] = static_cast<TestType>(0.25);
  owned.cache.points_changed();
  {
    const auto mesh = owned.mesh();
    CHECK(owned.cache.is_half_edges_fresh(mesh.geometry()));
    CHECK(owned.cache.is_face_link_fresh(mesh.geometry()));
    CHECK(owned.cache.is_vertex_link_fresh(mesh.geometry()));
    static_cast<void>(mesh.half_edges());
    static_cast<void>(mesh.face_link());
    static_cast<void>(mesh.vertex_link());
  }
  CHECK(owned.cache.half_edges_build_count() == 1);
  CHECK(owned.cache.face_link_build_count() == 1);
  CHECK(owned.cache.vertex_link_build_count() == 1);

  restate_points<prep_mixed_owned<TestType>, TestType>(
      owned, {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0, 3, 3, 0, 4, 4, 0});
  {
    const auto mesh = owned.mesh();
    CHECK_FALSE(owned.cache.is_half_edges_fresh(mesh.geometry()));
    CHECK(owned.cache.is_face_link_fresh(mesh.geometry()));
    CHECK_FALSE(owned.cache.is_vertex_link_fresh(mesh.geometry()));
    const auto &grown_half_edges = mesh.half_edges();
    const auto grown_vertex_link = mesh.vertex_link();
    CHECK(grown_half_edges.number_of_vertices() == 7);
    REQUIRE(grown_vertex_link.size() == 7);
    CHECK(grown_vertex_link[5].size() == 0);
    CHECK(grown_vertex_link[6].size() == 0);
  }
  CHECK(owned.cache.half_edges_build_count() == 2);
  CHECK(owned.cache.face_link_build_count() == 1);
  CHECK(owned.cache.vertex_link_build_count() == 2);

  restate_mixed_faces(owned, {0, 4, 7}, {0, 1, 3, 2, 2, 3, 4});
  {
    const auto mesh = owned.mesh();
    CHECK_FALSE(owned.cache.is_half_edges_fresh(mesh.geometry()));
    CHECK_FALSE(owned.cache.is_face_link_fresh(mesh.geometry()));
    CHECK_FALSE(owned.cache.is_vertex_link_fresh(mesh.geometry()));
    static_cast<void>(mesh.half_edges());
    static_cast<void>(mesh.face_link());
    static_cast<void>(mesh.vertex_link());
  }
  CHECK(owned.cache.half_edges_build_count() == 3);
  CHECK(owned.cache.face_link_build_count() == 2);
  CHECK(owned.cache.vertex_link_build_count() == 3);

  // THE DOOR: what a read refuses is a corner the points no longer reach, and
  // the cache is where that is asked, once per reading
  restate_points<prep_mixed_owned<TestType>, TestType>(owned,
                                                       {0, 0, 0, 1, 0, 0});
  {
    const auto mesh = owned.mesh();
    CHECK_THROWS_AS(mesh.half_edges(), std::out_of_range);
    CHECK_THROWS_AS(mesh.face_link(), std::out_of_range);
    CHECK_THROWS_AS(mesh.vertex_link(), std::out_of_range);
  }
  CHECK(owned.cache.half_edges_build_count() == 3);
  CHECK(owned.cache.face_link_build_count() == 2);
  CHECK(owned.cache.vertex_link_build_count() == 3);
}

TEMPLATE_TEST_CASE("an empty mixed mesh builds empty caches",
                   "[cpp][core][cache][carrier-integration]", float, double) {
  const prep_mixed_owned<TestType> owned{
      tf::cpp::test::polygons_of<prep_index, TestType>({0}, {}, {3, 4, 5})};
  const auto mesh = owned.mesh();

  tf::cpp::build_tree(mesh);
  tf::cpp::build_face_membership(mesh);
  const auto &half_edges = mesh.half_edges();
  const auto face_link = mesh.face_link();
  const auto vertex_link = mesh.vertex_link();
  REQUIRE(owned.cache.is_tree_fresh(mesh.geometry()));
  REQUIRE(owned.cache.is_face_membership_fresh(mesh.geometry()));
  REQUIRE(owned.cache.is_half_edges_fresh(mesh.geometry()));
  REQUIRE(owned.cache.is_face_link_fresh(mesh.geometry()));
  REQUIRE(owned.cache.is_vertex_link_fresh(mesh.geometry()));
  CHECK(mesh.tree().primitive_aabbs().empty());
  const auto membership = mesh.face_membership();
  REQUIRE(membership.size() == 1);
  CHECK(membership[0].size() == 0);
  CHECK(half_edges.number_of_faces() == 0);
  CHECK(half_edges.number_of_vertices() == 1);
  CHECK(half_edges.half_edges_buffer().size() == 0);
  CHECK(half_edges.face_half_edge_handles().empty());
  REQUIRE(half_edges.vertex_half_edge_handles().size() == 1);
  CHECK_FALSE(half_edges.vertex_half_edge_handles()[0].is_valid());
  CHECK(face_link.size() == 0);
  REQUIRE(vertex_link.size() == 1);
  CHECK(vertex_link[0].size() == 0);
  CHECK(owned.cache.tree_build_count() == 1);
  CHECK(owned.cache.face_membership_build_count() == 1);
}

TEMPLATE_TEST_CASE("a warm names the structures an entry reads, and only "
                   "rebuilds what a stated change staled",
                   "[cpp][core][cache]", float, double) {
  // an arrangement operand is `topology_form()`: tree, face membership and
  // manifold edge link. There is no verb for a query FAMILY — the caller names
  // the three facts, and a defensive call costs a stamp check
  auto owned = triangle_mesh<TestType>();
  tf::cpp::build_tree(owned.mesh());
  tf::cpp::build_face_membership(owned.mesh());
  tf::cpp::build_manifold_edge_link(owned.mesh());
  CHECK(owned.cache.tree_build_count() == 1);
  CHECK(owned.cache.face_membership_build_count() == 1);
  CHECK(owned.cache.manifold_edge_link_build_count() == 1);

  tf::cpp::build_tree(owned.mesh());
  tf::cpp::build_face_membership(owned.mesh());
  tf::cpp::build_manifold_edge_link(owned.mesh());
  CHECK(owned.cache.tree_build_count() == 1);
  CHECK(owned.cache.face_membership_build_count() == 1);
  CHECK(owned.cache.manifold_edge_link_build_count() == 1);

  owned.polygons.points_buffer().data_buffer()[0] = static_cast<TestType>(0.25);
  owned.cache.points_changed();
  CHECK_FALSE(owned.cache.is_tree_fresh(owned.mesh().geometry()));
  CHECK(owned.cache.is_face_membership_fresh(owned.mesh().geometry()));

  tf::cpp::build_tree(owned.mesh());
  tf::cpp::build_face_membership(owned.mesh());
  tf::cpp::build_manifold_edge_link(owned.mesh());
  CHECK(owned.cache.tree_build_count() == 2);
  CHECK(owned.cache.face_membership_build_count() == 1);
  CHECK(owned.cache.manifold_edge_link_build_count() == 1);
}

TEMPLATE_TEST_CASE("a normal is where the points stand, so a stated move "
                   "stales both and leaves the connectivity alone",
                   "[cpp][core][cache]", float, double) {
  auto owned = triangle_mesh<TestType>();
  tf::cpp::build_point_normals(owned.mesh());
  CHECK(owned.cache.face_normals_build_count() == 1);
  CHECK(owned.cache.point_normals_build_count() == 1);
  CHECK(owned.cache.face_membership_build_count() == 1);

  owned.polygons.points_buffer().data_buffer()[0] = static_cast<TestType>(0.25);
  owned.cache.points_changed();
  CHECK_FALSE(owned.cache.is_face_normals_fresh(owned.mesh().geometry()));
  CHECK_FALSE(owned.cache.is_point_normals_fresh(owned.mesh().geometry()));
  CHECK(owned.cache.is_face_membership_fresh(owned.mesh().geometry()));

  tf::cpp::build_point_normals(owned.mesh());
  CHECK(owned.cache.face_normals_build_count() == 2);
  CHECK(owned.cache.point_normals_build_count() == 2);
  CHECK(owned.cache.face_membership_build_count() == 1);

  // both read the connectivity too — one for the corners it spans, one for the
  // faces it gathers — so a restated face stales both of them as well
  auto &corners = owned.polygons.faces_buffer().data_buffer();
  const auto first = corners[1];
  corners[1] = corners[2];
  corners[2] = first;
  owned.cache.faces_changed();
  CHECK_FALSE(owned.cache.is_face_normals_fresh(owned.mesh().geometry()));
  CHECK_FALSE(owned.cache.is_point_normals_fresh(owned.mesh().geometry()));
}

TEST_CASE("a verb builds what its structure stands on", "[cpp][core][cache]") {
  // a verb is idempotent and builds its dependencies, which is what makes
  // naming the top-level facts enough
  const auto owned = triangle_mesh<float>();
  const auto mesh = owned.mesh();
  tf::cpp::build_face_link(mesh);
  CHECK(owned.cache.is_face_link_fresh(mesh.geometry()));
  CHECK(owned.cache.is_face_membership_fresh(mesh.geometry()));
  CHECK(owned.cache.face_membership_build_count() == 1);

  tf::cpp::build_face_link(mesh);
  tf::cpp::build_face_membership(mesh);
  CHECK(owned.cache.face_link_build_count() == 1);
  CHECK(owned.cache.face_membership_build_count() == 1);

  // a point normal averages the face normals around it, so its verb states
  // those and the membership it gathers them through
  tf::cpp::build_point_normals(mesh);
  CHECK(owned.cache.is_point_normals_fresh(mesh.geometry()));
  CHECK(owned.cache.is_face_normals_fresh(mesh.geometry()));
  CHECK(owned.cache.face_normals_build_count() == 1);
  CHECK(owned.cache.point_normals_build_count() == 1);

  tf::cpp::build_point_normals(mesh);
  tf::cpp::build_face_normals(mesh);
  CHECK(owned.cache.face_normals_build_count() == 1);
  CHECK(owned.cache.point_normals_build_count() == 1);

  // an empty cloud has a tree of nothing, which is a tree
  const tf::cpp::test::owned_point_cloud<float> nothing;
  const auto empty = nothing.point_cloud();
  CHECK_NOTHROW(tf::cpp::build_tree(empty));
  CHECK(nothing.cache.is_tree_fresh(empty.geometry()));
}

TEST_CASE("a cache is warmed before it is shared, and read freely after",
          "[cpp][core][cache]") {
  // THE CONTRACT: filling is not thread safe, so the verbs run here, on one
  // thread; what the workers then do is read filled state, which is free and
  // unlimited and rebuilds nothing. EVERY ask is in this reader, because every
  // one of them fills, and a warm one must write nothing at all — no structure,
  // no stamp, and not the fill guard's own flag.
  // THE UNIT OF SHARING is why each worker assembles its OWN mesh over the
  // shared geometry and cache: a mesh keeps what it was handed, so one value
  // read by eight threads would race on its own slots with the cache untouched
  const auto owned = cache_race_grid_mesh(64);
  tf::cpp::build_tree(owned.mesh());
  tf::cpp::build_half_edges(owned.mesh());
  tf::cpp::build_manifold_edge_link(owned.mesh());
  tf::cpp::build_face_link(owned.mesh());
  tf::cpp::build_vertex_link(owned.mesh());
  owned.mesh().require_indices();
  REQUIRE(owned.cache.tree_build_count() == 1);
  REQUIRE(owned.cache.half_edges_build_count() == 1);
  REQUIRE(owned.cache.face_membership_build_count() == 1);
  REQUIRE(owned.cache.manifold_edge_link_build_count() == 1);
  REQUIRE(owned.cache.face_link_build_count() == 1);
  REQUIRE(owned.cache.vertex_link_build_count() == 1);

  std::vector<std::future<std::size_t>> readers;
  readers.reserve(8);
  for (int worker = 0; worker != 8; ++worker)
    readers.push_back(std::async(std::launch::async, [&owned]() -> std::size_t {
      const auto mesh = owned.mesh();
      mesh.require_indices();
      return mesh.manifold_edge_link().size() + mesh.face_membership().size() +
             mesh.face_link().size() + mesh.vertex_link().size() +
             static_cast<std::size_t>(mesh.half_edges().n_faces()) +
             static_cast<std::size_t>(mesh.tree().bv().max[0]);
    }));
  const auto first = readers.front().get();
  for (std::size_t i = 1; i != readers.size(); ++i)
    CHECK(readers[i].get() == first);

  CHECK(owned.cache.tree_build_count() == 1);
  CHECK(owned.cache.half_edges_build_count() == 1);
  CHECK(owned.cache.face_membership_build_count() == 1);
  CHECK(owned.cache.manifold_edge_link_build_count() == 1);
  CHECK(owned.cache.face_link_build_count() == 1);
  CHECK(owned.cache.vertex_link_build_count() == 1);
}

#ifndef NDEBUG
TEST_CASE("the fill guard holds the flag a concurrent fill would find",
          "[cpp][core][cache]") {
  // the detector's own assert cannot be exercised without aborting, so what a
  // fixture can state is the flag discipline underneath it; a release build
  // never takes the flag at all, which is why this case is not compiled there
  std::atomic<bool> filling{false};
  {
    tf::cpp::detail::fill_guard guard(filling);
    CHECK(filling.load());
  }
  CHECK_FALSE(filling.load());
}
#endif

TEST_CASE("N instances over one geometry and one cache warm it once",
          "[cpp][core][cache]") {
  // THE INSTANCING SHAPE the assembly advertises: N meshes, one geometry, one
  // cache, a frame each. Warming them is warming ONE cache, so the width of a
  // warm is the DISTINCT caches and the operand count is not a proxy for it —
  // the second instance's ask is answered from what the first filled, and no
  // second fill is ever engaged. A debug build is where the guard states that.
  const auto owned = cache_race_grid_mesh(32);
  using instance_t = tf::cpp::mesh<prep_index, float>;
  std::vector<instance_t> instances;
  instances.reserve(4);
  for (int instance = 0; instance != 4; ++instance)
    instances.push_back(owned.mesh());

  for (const auto &value : instances) {
    tf::cpp::build_tree(value);
    tf::cpp::build_face_membership(value);
    tf::cpp::build_manifold_edge_link(value);
  }

  CHECK(owned.cache.tree_build_count() == 1);
  CHECK(owned.cache.face_membership_build_count() == 1);
  CHECK(owned.cache.manifold_edge_link_build_count() == 1);

  const auto first = instances.front().tree().bv().max[0];
  for (const auto &value : instances) {
    CHECK(value.tree().bv().max[0] == first);
    CHECK(value.face_membership().size() == owned.mesh().number_of_points());
    CHECK(value.manifold_edge_link().size() == owned.mesh().number_of_faces());
  }
  CHECK(owned.cache.tree_build_count() == 1);
}

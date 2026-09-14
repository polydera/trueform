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
#include "nd_array.hpp"

#include "trueform/cpp/core/build_edge_membership.hpp"
#include "trueform/cpp/core/build_tree.hpp"
#include "trueform/cpp/core/build_vertex_link.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/edge_mesh_cache.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

template <typename Real, std::size_t Dims>
auto edge_points4() -> std::vector<Real> {
  if constexpr (Dims == 2)
    return {0, 0, 2, 0, 2, 3, -1, 0};
  else
    return {0, 0, 0, 2, 0, 0, 2, 3, 1, -1, 0, 4};
}

template <typename Real, std::size_t Dims>
auto edge_points5() -> std::vector<Real> {
  if constexpr (Dims == 2)
    return {0, 0, 2, 0, 2, 3, -1, 0, 5, 5};
  else
    return {0, 0, 0, 2, 0, 0, 2, 3, 1, -1, 0, 4, 5, 5, 5};
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_owned() -> tf::cpp::test::owned_edge_mesh<Index, Real, Dims> {
  const std::vector<Index> edges{0, 1, 1, 2, 1, 3};
  return {tf::cpp::test::segments_of<Index, Real, Dims>(
      edges, edge_points4<Real, Dims>())};
}

template <typename Real, std::size_t Dims>
auto edge_identity() -> std::array<Real, (Dims + 1) * (Dims + 1)> {
  std::array<Real, (Dims + 1) * (Dims + 1)> values{};
  for (std::size_t row = 0; row != Dims + 1; ++row)
    values[row * (Dims + 1) + row] = Real{1};
  return values;
}

template <typename Index>
auto offset_cache(std::initializer_list<Index> offsets,
                  std::initializer_list<Index> data)
    -> tf::cpp::offset_blocked_buffer<Index, Index> {
  return tf::cpp::offset_blocked_buffer<Index, Index>::create(
      tf::cpp::test::make_nd_array<Index>(offsets,
                                          {static_cast<int>(offsets.size())}),
      tf::cpp::test::make_nd_array<Index>(data,
                                          {static_cast<int>(data.size())}));
}

/// The caller's own storage, restated, followed by the statement that says so.
template <typename Owned, typename Real>
auto restate_points(Owned &owned, const std::vector<Real> &coordinates)
    -> void {
  auto &storage = owned.segments.points_buffer().data_buffer();
  storage.allocate(coordinates.size());
  std::copy(coordinates.begin(), coordinates.end(), storage.begin());
  owned.cache.points_changed();
}

template <typename Index, typename Real, std::size_t Dims>
auto check_matrix_combination() -> void {
  INFO("Real bytes = " << sizeof(Real) << ", Index bytes = " << sizeof(Index)
                       << ", Dims = " << Dims);
  auto owned = edge_owned<Index, Real, Dims>();
  const auto edge_mesh = owned.edge_mesh();

  using edge_type = std::decay_t<decltype(edge_mesh.edges()[0])>;
  STATIC_REQUIRE(std::tuple_size<edge_type>::value == 2);

  CHECK(edge_mesh.number_of_edges() == 3);
  CHECK(edge_mesh.number_of_points() == 4);
  CHECK(edge_mesh.segments().size() == 3);
  CHECK(edge_mesh.segments()[1][1][Dims - 1] ==
        (Dims == 2 ? Real{3} : Real{1}));

  tf::cpp::build_tree(edge_mesh);
  tf::cpp::build_edge_membership(edge_mesh);
  tf::cpp::build_vertex_link(edge_mesh);
  STATIC_REQUIRE(std::is_const_v<std::remove_reference_t<
                     decltype(edge_mesh.edge_membership()[0][0])>>);
  STATIC_REQUIRE(
      std::is_const_v<
          std::remove_reference_t<decltype(edge_mesh.vertex_link()[0][0])>>);
  REQUIRE(owned.cache.is_tree_fresh(edge_mesh.geometry()));
  REQUIRE(owned.cache.is_edge_membership_fresh(edge_mesh.geometry()));
  REQUIRE(owned.cache.is_vertex_link_fresh(edge_mesh.geometry()));
  CHECK(edge_mesh.tree().ids().size() == 3);

  const auto membership = edge_mesh.edge_membership();
  REQUIRE(membership.size() == 4);
  CHECK(membership[0].size() == 1);
  CHECK(membership[0][0] == Index{0});
  CHECK(membership[1].size() == 3);
  CHECK(membership[2][0] == Index{1});
  CHECK(membership[3][0] == Index{2});

  const auto link = edge_mesh.vertex_link();
  REQUIRE(link.size() == 4);
  CHECK(link[0][0] == Index{1});
  CHECK(link[1].size() == 3);
  CHECK(link[2][0] == Index{1});
  CHECK(link[3][0] == Index{1});

  const auto plain_form = edge_mesh.form();
  STATIC_REQUIRE(std::is_const_v<
                 std::remove_reference_t<decltype(plain_form.edges()[0][0])>>);
  STATIC_REQUIRE(std::is_const_v<
                 std::remove_reference_t<decltype(plain_form.points()[0][0])>>);
  CHECK(plain_form.size() == 3);
  // the frame is always tagged, identity when none was given
  STATIC_REQUIRE(tf::has_frame_policy<decltype(plain_form)>);
  CHECK(plain_form.transformation()(0, 0) == Real{1});
  CHECK(edge_mesh.topology_form().edge_membership().size() == 4);

  owned.place(edge_identity<Real, Dims>());
  const auto builds = owned.cache.tree_build_count();
  const auto framed = owned.edge_mesh();
  const auto framed_form = framed.form();
  CHECK(framed_form.size() == 3);
  CHECK(framed_form[0][0][0] == Real{0});
  // the placement is not part of the reading, so it rebuilds nothing
  CHECK(owned.cache.tree_build_count() == builds);
}

using default_edge_mesh = tf::cpp::edge_mesh<tf::cpp::default_index_t, float>;
using explicit_default_edge_mesh =
    tf::cpp::edge_mesh<tf::cpp::default_index_t, float, 3>;
using wide_2d_edge_mesh = tf::cpp::edge_mesh<std::int64_t, double, 2>;
using wide_3d_edge_mesh = tf::cpp::edge_mesh<std::int64_t, double, 3>;

static_assert(std::is_same_v<default_edge_mesh, explicit_default_edge_mesh>);
static_assert(
    std::is_same_v<tf::cpp::edge_mesh_cache<std::int64_t, double>,
                   tf::cpp::edge_mesh_cache<std::int64_t, double, 3>>);
static_assert(!std::is_default_constructible_v<wide_2d_edge_mesh>);
static_assert(!std::is_same_v<
              decltype(std::declval<const wide_2d_edge_mesh &>().tree()),
              decltype(std::declval<const wide_3d_edge_mesh &>().tree())>);

} // namespace

TEST_CASE("edge mesh public carrier matrix preserves native storage and forms",
          "[cpp][core][edge-mesh][matrix]") {
  check_matrix_combination<std::int32_t, float, 2>();
  check_matrix_combination<std::int32_t, float, 3>();
  check_matrix_combination<std::int64_t, float, 2>();
  check_matrix_combination<std::int64_t, float, 3>();
  check_matrix_combination<std::int32_t, double, 2>();
  check_matrix_combination<std::int32_t, double, 3>();
  check_matrix_combination<std::int64_t, double, 2>();
  check_matrix_combination<std::int64_t, double, 3>();
}

TEST_CASE("edge mesh generations invalidate only dependent authorities",
          "[cpp][core][edge-mesh][cache][mutation]") {
  auto owned = edge_owned<tf::cpp::default_index_t, float, 3>();
  CHECK(owned.edge_mesh().form().size() == 3);
  CHECK_FALSE(owned.cache.is_edge_membership_built());
  CHECK(owned.edge_mesh().topology_form().edge_membership().size() == 4);
  CHECK(owned.cache.is_edge_membership_fresh(owned.edge_mesh().geometry()));
  tf::cpp::build_tree(owned.edge_mesh());
  tf::cpp::build_vertex_link(owned.edge_mesh());
  CHECK(owned.cache.tree_build_count() == 1);
  CHECK(owned.cache.edge_membership_build_count() == 1);
  CHECK(owned.cache.vertex_link_build_count() == 1);

  // THE TWO STAMPS: a point that moved stales the tree and leaves the
  // structures that only count points where they were
  const auto edges_generation = owned.cache.edges_generation();
  const auto points_generation = owned.cache.points_generation();
  owned.segments.points_buffer().data_buffer()[0] = 7;
  owned.cache.points_changed();
  CHECK(owned.cache.edges_generation() == edges_generation);
  CHECK(owned.cache.points_generation() == points_generation + 1);
  {
    const auto moved = owned.edge_mesh();
    CHECK_FALSE(owned.cache.is_tree_fresh(moved.geometry()));
    CHECK(owned.cache.is_edge_membership_fresh(moved.geometry()));
    CHECK(owned.cache.is_vertex_link_fresh(moved.geometry()));
    tf::cpp::build_tree(moved);
  }
  CHECK(owned.cache.tree_build_count() == 2);

  // a point set of another SIZE is a different reading for every structure
  restate_points<decltype(owned), float>(owned, edge_points5<float, 3>());
  {
    const auto grown = owned.edge_mesh();
    CHECK_FALSE(owned.cache.is_tree_fresh(grown.geometry()));
    CHECK_FALSE(owned.cache.is_edge_membership_fresh(grown.geometry()));
    CHECK_FALSE(owned.cache.is_vertex_link_fresh(grown.geometry()));
    tf::cpp::build_edge_membership(grown);
    tf::cpp::build_vertex_link(grown);
    CHECK(grown.edge_membership().size() == 5);
    CHECK(grown.vertex_link().size() == 5);
  }

  // restated edges stale everything that walks them
  owned.segments.edges_buffer().data_buffer()[0] = 4;
  owned.cache.edges_changed();
  {
    const auto restated = owned.edge_mesh();
    CHECK_FALSE(owned.cache.is_tree_fresh(restated.geometry()));
    CHECK_FALSE(owned.cache.is_edge_membership_fresh(restated.geometry()));
    CHECK_FALSE(owned.cache.is_vertex_link_fresh(restated.geometry()));
    tf::cpp::build_tree(restated);
    tf::cpp::build_edge_membership(restated);
    tf::cpp::build_vertex_link(restated);
    CHECK(restated.edge_membership()[4][0] == 0);
  }

  // the placement is not part of a reading, so stating one rebuilds nothing
  const auto builds = owned.cache.tree_build_count();
  owned.place(edge_identity<float, 3>());
  const auto placed = owned.edge_mesh();
  CHECK(owned.cache.is_tree_fresh(placed.geometry()));
  CHECK(owned.cache.is_edge_membership_fresh(placed.geometry()));
  CHECK(owned.cache.is_vertex_link_fresh(placed.geometry()));
  CHECK(owned.cache.tree_build_count() == builds);
}

TEST_CASE("a copied edge cache shares nothing with the one it came from",
          "[cpp][core][edge-mesh][ownership]") {
  auto owned = edge_owned<std::int64_t, double, 2>();
  tf::cpp::build_tree(owned.edge_mesh());
  tf::cpp::build_edge_membership(owned.edge_mesh());
  tf::cpp::build_vertex_link(owned.edge_mesh());

  auto copy = owned.cache;
  const auto &stored = owned.segments;
  const auto over_copy = tf::cpp::edge_mesh<std::int64_t, double, 2>(
      stored.edges(), stored.points(), copy);
  CHECK(copy.is_edge_membership_fresh(over_copy.geometry()));
  CHECK_FALSE(copy.is_tree_built());
  // it built none of what it carries
  CHECK(copy.tree_build_count() == 0);
  CHECK(copy.edge_membership_build_count() == 0);
  CHECK(copy.vertex_link_build_count() == 0);
  static_cast<void>(over_copy.tree());
  CHECK(copy.tree_build_count() == 1);
  CHECK(owned.cache.tree_build_count() == 1);
}

TEST_CASE("edge mesh accepts coherent empty storage and empty caches",
          "[cpp][core][edge-mesh][empty]") {
  const tf::cpp::test::owned_edge_mesh<std::int32_t, float, 2> owned;
  const auto empty = owned.edge_mesh();
  CHECK(empty.number_of_edges() == 0);
  CHECK(empty.number_of_points() == 0);
  CHECK(empty.edges().size() == 0);
  CHECK(empty.points().size() == 0);
  CHECK(empty.segments().size() == 0);

  tf::cpp::build_tree(empty);
  tf::cpp::build_edge_membership(empty);
  tf::cpp::build_vertex_link(empty);
  CHECK(owned.cache.is_tree_fresh(empty.geometry()));
  CHECK(empty.tree().nodes().size() == 0);
  CHECK(empty.edge_membership().size() == 0);
  CHECK(empty.vertex_link().size() == 0);
  CHECK(empty.form().size() == 0);
  CHECK(empty.topology_form().edge_membership().size() == 0);
}

TEST_CASE("edge mesh refuses indices and malformed stated topology",
          "[cpp][core][edge-mesh][validation]") {
  auto owned = edge_owned<tf::cpp::default_index_t, float, 3>();
  const auto edge_mesh = owned.edge_mesh();

  CHECK_THROWS_AS(
      owned.cache.set_edge_membership(offset_cache<std::int32_t>({0, 1}, {0}),
                                      edge_mesh.geometry()),
      std::invalid_argument);
  CHECK_THROWS_AS(owned.cache.set_edge_membership(
                      offset_cache<std::int32_t>({0, 1, 1, 1, 1}, {3}),
                      edge_mesh.geometry()),
                  std::out_of_range);
  CHECK_THROWS_AS(owned.cache.set_vertex_link(
                      offset_cache<std::int32_t>({0, 1, 1, 1, 1}, {4}),
                      edge_mesh.geometry()),
                  std::out_of_range);

  owned.cache.set_edge_membership(
      offset_cache<std::int32_t>({0, 1, 4, 5, 6}, {0, 0, 1, 2, 1, 2}),
      edge_mesh.geometry());
  CHECK(owned.cache.is_edge_membership_fresh(edge_mesh.geometry()));
  CHECK(edge_mesh.edge_membership()[0][0] == 0);

  owned.cache.set_vertex_link(
      offset_cache<std::int32_t>({0, 1, 4, 5, 6}, {1, 0, 2, 3, 1, 1}),
      edge_mesh.geometry());
  CHECK(owned.cache.is_vertex_link_fresh(edge_mesh.geometry()));

  // an edge naming a point the reading does not have is refused at the door,
  // which is the cache's, once per reading
  const std::vector<tf::cpp::default_index_t> absent_edges{0, 4};
  const auto absent =
      tf::cpp::test::segments_of<tf::cpp::default_index_t, float, 3>(
          absent_edges, edge_points4<float, 3>());
  tf::cpp::edge_mesh_cache<tf::cpp::default_index_t, float, 3> cache;
  const auto refused = tf::cpp::edge_mesh<tf::cpp::default_index_t, float, 3>(
      absent.edges(), absent.points(), cache);
  CHECK_THROWS_AS(refused.require_indices(), std::out_of_range);
}

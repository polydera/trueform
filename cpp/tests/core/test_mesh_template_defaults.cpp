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

#include "trueform/core/static_size.hpp"
#include "trueform/cpp/core/build_face_membership.hpp"
#include "trueform/cpp/core/build_tree.hpp"
#include "trueform/cpp/core/cache.hpp"
#include "trueform/cpp/core/mesh.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>

namespace {

template <typename Real, std::size_t Dims>
auto defaults_identity() -> std::array<Real, (Dims + 1) * (Dims + 1)> {
  std::array<Real, (Dims + 1) * (Dims + 1)> values{};
  for (std::size_t row = 0; row != Dims + 1; ++row)
    values[row * (Dims + 1) + row] = Real{1};
  return values;
}

template <typename Index, typename Real, std::size_t Dims>
auto check_mesh_carrier_matrix_combination() -> void {
  INFO("Real bytes = " << sizeof(Real) << ", Index bytes = " << sizeof(Index)
                       << ", Dims = " << Dims);
  using owned_t = tf::cpp::test::owned_mesh<Index, Real, Dims>;

  auto owned = [] {
    if constexpr (Dims == 2)
      return owned_t{tf::cpp::test::polygons_of<Index, Real, 2>(
          {0, 1, 2}, {0, 0, 2, 0, 0, 3})};
    else
      return owned_t{tf::cpp::test::polygons_of<Index, Real, 3>(
          {0, 1, 2}, {0, 0, 0, 2, 0, 0, 0, 3, 4})};
  }();

  const auto mesh = owned.mesh();
  CHECK(mesh.number_of_faces() == 1);
  CHECK(mesh.number_of_points() == 3);
  CHECK(mesh.faces()[0][2] == Index{2});
  CHECK(mesh.faces()[0].size() == 3);

  // THE FRAME IS ALWAYS THERE: a mesh assembled without one is placed by
  // identity, which is what every body reads it at
  CHECK(mesh.frame()(0, 0) == Real{1});
  CHECK(mesh.frame()(0, static_cast<int>(Dims)) == Real{0});
  owned.place(defaults_identity<Real, Dims>());
  CHECK(owned.mesh().frame()(0, 0) == Real{1});

  tf::cpp::build_tree(mesh);
  tf::cpp::build_face_membership(mesh);
  REQUIRE(owned.cache.is_tree_fresh(mesh.geometry()));
  REQUIRE(owned.cache.is_face_membership_fresh(mesh.geometry()));
  CHECK(mesh.tree().bv().max[Dims - 1] == (Dims == 2 ? Real{3} : Real{4}));

  const auto membership = mesh.face_membership();
  REQUIRE(membership.size() == 3);
  CHECK(membership[2][0] == Index{0});

  // A CACHE IS A VALUE: a copy shares nothing, so it keeps what it can own
  // outright and builds a tree of its own when one is asked for. It built
  // none of what it carries, so every count on it starts at zero
  auto copy = owned.cache;
  CHECK(copy.is_face_membership_fresh(mesh.geometry()));
  CHECK_FALSE(copy.is_tree_built());
  CHECK_FALSE(copy.is_winding_moments_built());
  CHECK_FALSE(copy.is_half_edges_built());
  CHECK(copy.tree_build_count() == 0);
  CHECK(copy.winding_moments_build_count() == 0);
  CHECK(copy.half_edges_build_count() == 0);
  CHECK(copy.face_membership_build_count() == 0);
  CHECK(copy.manifold_edge_link_build_count() == 0);
  CHECK(copy.face_link_build_count() == 0);
  CHECK(copy.vertex_link_build_count() == 0);
  const auto &stored = owned.polygons;
  const auto over_copy =
      tf::cpp::mesh<Index, Real, Dims>(stored.faces(), stored.points(), copy);
  static_cast<void>(over_copy.tree());
  static_cast<void>(over_copy.half_edges());
  CHECK(copy.is_half_edges_built());
  CHECK(copy.tree_build_count() == 1);
  CHECK(copy.face_membership_build_count() == 0);
  CHECK(owned.cache.tree_build_count() == 1);
  CHECK_FALSE(owned.cache.is_half_edges_built());
}

using default_mesh = tf::cpp::mesh<tf::cpp::default_index_t, float>;
using explicit_dims_mesh = tf::cpp::mesh<tf::cpp::default_index_t, float, 3>;
using explicit_triangle_mesh =
    tf::cpp::mesh<tf::cpp::default_index_t, float, 3, 3>;
using wide_2d_mesh = tf::cpp::mesh<std::int64_t, double, 2>;
using wide_2d_mixed_mesh =
    tf::cpp::mesh<std::int64_t, double, 2, tf::dynamic_size>;
using wide_3d_mesh = tf::cpp::mesh<std::int64_t, double, 3>;

static_assert(std::is_same_v<default_mesh, explicit_dims_mesh>);
static_assert(std::is_same_v<default_mesh, explicit_triangle_mesh>);
static_assert(
    std::is_same_v<tf::cpp::cache<tf::cpp::default_index_t, float>,
                   tf::cpp::cache<tf::cpp::default_index_t, float, 3, 3>>);
static_assert(
    std::is_same_v<typename default_mesh::cache_type,
                   tf::cpp::cache<tf::cpp::default_index_t, float, 3, 3>>);

// the layout is the geometry's own type, so the faces a mesh reads are the
// blocks that layout states and no other
static_assert(!std::is_same_v<
              decltype(std::declval<const wide_2d_mesh &>().faces()),
              decltype(std::declval<const wide_2d_mixed_mesh &>().faces())>);
static_assert(
    !std::is_same_v<decltype(std::declval<const wide_2d_mesh &>().points()),
                    decltype(std::declval<const wide_3d_mesh &>().points())>);
static_assert(
    !std::is_same_v<decltype(std::declval<const wide_2d_mesh &>().polygons()),
                    decltype(std::declval<const wide_3d_mesh &>().polygons())>);
static_assert(
    !std::is_same_v<decltype(std::declval<const wide_2d_mesh &>().tree()),
                    decltype(std::declval<const wide_3d_mesh &>().tree())>);
static_assert(!std::is_default_constructible_v<wide_2d_mesh>);
static_assert(std::is_copy_constructible_v<wide_2d_mesh>);

// a triangle mesh's face keeps its static arity, which is what the whole
// arity axis is for
using triangle_face_type =
    std::decay_t<decltype(std::declval<const default_mesh &>().faces()[0])>;
static_assert(std::tuple_size<triangle_face_type>::value == 3,
              "triangle mesh ranges must retain static arity");

} // namespace

TEST_CASE("approved mesh carrier matrix links and preserves native shapes",
          "[cpp][core][mesh][templates][matrix]") {
  check_mesh_carrier_matrix_combination<std::int32_t, float, 2>();
  check_mesh_carrier_matrix_combination<std::int32_t, float, 3>();
  check_mesh_carrier_matrix_combination<std::int64_t, float, 2>();
  check_mesh_carrier_matrix_combination<std::int64_t, float, 3>();
  check_mesh_carrier_matrix_combination<std::int32_t, double, 2>();
  check_mesh_carrier_matrix_combination<std::int32_t, double, 3>();
  check_mesh_carrier_matrix_combination<std::int64_t, double, 2>();
  check_mesh_carrier_matrix_combination<std::int64_t, double, 3>();
}

TEST_CASE("a restated geometry is a reading of its own",
          "[cpp][core][mesh][templates]") {
  auto owned = tf::cpp::test::owned_mesh<tf::cpp::default_index_t, float>{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, float>(
          {0, 1, 2}, {0, 0, 0, 1, 0, 0, 0, 1, 0})};

  CHECK(owned.mesh().faces()[0][0] == 0);
  tf::cpp::build_tree(owned.mesh());
  tf::cpp::build_face_membership(owned.mesh());
  CHECK(owned.cache.tree_build_count() == 1);

  // the caller writes through its own storage and states what changed; the
  // next reading is what the cache answers for
  auto &corners = owned.polygons.faces_buffer().data_buffer();
  corners[0] = 2;
  corners[2] = 0;
  owned.cache.faces_changed();
  const auto restated = owned.mesh();
  CHECK(restated.faces()[0][0] == 2);
  CHECK(restated.faces()[0].size() == 3);
  CHECK(restated.polygons()[0].size() == 3);
  CHECK_FALSE(owned.cache.is_tree_fresh(restated.geometry()));
  CHECK_FALSE(owned.cache.is_face_membership_fresh(restated.geometry()));
}

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
#include "mixed_mesh.hpp"
#include "nd_array.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/build_face_membership.hpp"
#include "trueform/cpp/core/build_manifold_edge_link.hpp"
#include "trueform/cpp/core/build_vertex_link.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/topology.hpp"
#include "trueform/cpp/topology/cdt.hpp"
#include "trueform/cpp/topology/domain_labels.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename T>
auto make_array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(values.size());
  auto output = buffer.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename T, typename U>
auto equals(const tf::cpp::nd_array<T> &actual,
            std::initializer_list<U> expected) -> bool {
  if (actual.length() != expected.size())
    return false;
  return std::equal(actual.begin(), actual.end(), expected.begin());
}

template <typename T>
auto same_array(const tf::cpp::nd_array<T> &a, const tf::cpp::nd_array<T> &b)
    -> bool {
  return a.raw_shape() == b.raw_shape() &&
         std::equal(a.begin(), a.end(), b.begin());
}

template <typename Connectivity, typename = void>
struct supports_connected_component_labels : std::false_type {};

template <typename Connectivity>
struct supports_connected_component_labels<
    Connectivity, std::void_t<decltype(tf::cpp::label_connected_components(
                      std::declval<const Connectivity &>()))>>
    : std::true_type {};

template <typename Edges, typename = void>
struct supports_connect_edges_to_paths : std::false_type {};

template <typename Edges>
struct supports_connect_edges_to_paths<
    Edges, std::void_t<decltype(tf::cpp::connect_edges_to_paths(
               std::declval<const Edges &>()))>> : std::true_type {};

template <typename Index>
auto canonicalize_path(std::vector<Index> path) -> std::vector<Index> {
  const auto closed = path.size() > 1 && path.front() == path.back();
  if (closed)
    path.pop_back();
  if (path.empty())
    return path;

  auto reversed = path;
  std::reverse(reversed.begin(), reversed.end());
  if (!closed)
    return std::min(path, reversed);

  auto best = path;
  auto consider_rotations = [&](const std::vector<Index> &candidate) {
    for (std::size_t shift = 0; shift < candidate.size(); ++shift) {
      std::vector<Index> rotated;
      rotated.reserve(candidate.size());
      rotated.insert(rotated.end(), candidate.begin() + shift, candidate.end());
      rotated.insert(rotated.end(), candidate.begin(),
                     candidate.begin() + shift);
      if (rotated < best)
        best = std::move(rotated);
    }
  };
  consider_rotations(path);
  consider_rotations(reversed);
  return best;
}

template <typename Index>
auto canonicalize_paths(
    const tf::cpp::offset_blocked_buffer<Index, Index> &paths)
    -> std::vector<std::vector<Index>> {
  std::vector<std::vector<Index>> result;
  result.reserve(static_cast<std::size_t>(paths.size()));
  for (int path_id = 0; path_id < paths.size(); ++path_id) {
    const auto path = paths.get(path_id);
    result.push_back(
        canonicalize_path(std::vector<Index>(path.begin(), path.end())));
  }
  std::sort(result.begin(), result.end());
  return result;
}

template <typename Index>
auto has_valid_offsets(
    const tf::cpp::offset_blocked_buffer<Index, Index> &paths) -> bool {
  if (!paths.is_valid())
    return false;
  const auto offsets = paths.offsets();
  const auto data = paths.data();
  if (offsets.ndim() != 1 || data.ndim() != 1 || offsets.empty() ||
      offsets.length() != static_cast<std::size_t>(paths.size() + 1) ||
      offsets[0] != Index{0} ||
      offsets[offsets.length() - 1] != static_cast<Index>(data.length()))
    return false;
  return std::is_sorted(offsets.begin(), offsets.end());
}

auto block_equals(
    const tf::cpp::offset_blocked_buffer<std::int32_t, std::int32_t> &blocks,
    int index, std::initializer_list<std::int32_t> expected) -> bool {
  auto actual = blocks.get(index);
  std::vector<std::int32_t> actual_values(actual.begin(), actual.end());
  std::vector<std::int32_t> expected_values(expected);
  std::sort(actual_values.begin(), actual_values.end());
  std::sort(expected_values.begin(), expected_values.end());
  return actual_values == expected_values;
}

auto has_consistent_shared_edges(const tf::cpp::nd_array<std::int32_t> &faces)
    -> bool {
  std::map<std::pair<std::int32_t, std::int32_t>, std::pair<int, int>> edges;
  for (int face = 0; face < faces.shape_at(0); ++face) {
    for (int edge = 0; edge < 3; ++edge) {
      const auto first = faces[static_cast<std::size_t>(face * 3 + edge)];
      const auto second =
          faces[static_cast<std::size_t>(face * 3 + (edge + 1) % 3)];
      const auto key =
          std::make_pair(std::min(first, second), std::max(first, second));
      auto &entry = edges[key];
      ++entry.first;
      entry.second += first < second ? 1 : -1;
    }
  }
  for (const auto &entry : edges)
    if (entry.second.first == 2 && entry.second.second != 0)
      return false;
  return true;
}

auto same_face_vertices(const tf::cpp::nd_array<std::int32_t> &a,
                        const tf::cpp::nd_array<std::int32_t> &b) -> bool {
  if (a.raw_shape() != b.raw_shape())
    return false;
  for (int face = 0; face < a.shape_at(0); ++face) {
    std::array<std::int32_t, 3> a_face{
        a[static_cast<std::size_t>(face * 3)],
        a[static_cast<std::size_t>(face * 3 + 1)],
        a[static_cast<std::size_t>(face * 3 + 2)]};
    std::array<std::int32_t, 3> b_face{
        b[static_cast<std::size_t>(face * 3)],
        b[static_cast<std::size_t>(face * 3 + 1)],
        b[static_cast<std::size_t>(face * 3 + 2)]};
    std::sort(a_face.begin(), a_face.end());
    std::sort(b_face.begin(), b_face.end());
    if (a_face != b_face)
      return false;
  }
  return true;
}

template <typename Real>
using parity_mesh = tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real>;

template <typename Real>
auto two_triangles(bool inconsistent = false) -> parity_mesh<Real> {
  using Index = tf::cpp::default_index_t;
  return {tf::cpp::test::polygons_of<Index, Real>(
      inconsistent ? std::initializer_list<Index>{0, 1, 2, 1, 2, 3}
                   : std::initializer_list<Index>{0, 1, 2, 1, 3, 2},
      {0, 0, 0, 1, 0, 0, Real{0.5}, 1, 0, Real{1.5}, 1, 0})};
}

template <typename Real> auto single_triangle() -> parity_mesh<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2}, {0, 0, 0, 1, 0, 0, Real{0.5}, 1, 0})};
}

template <typename Real> auto tetrahedron() -> parity_mesh<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2},
      {0, 0, 0, 1, 0, 0, Real{0.5}, 1, 0, Real{0.5}, Real{0.5}, 1})};
}

template <typename Real> auto non_manifold_mesh() -> parity_mesh<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 0, 1, 3, 0, 1, 4},
      {0, 0, 0, 1, 0, 0, Real{0.5}, 1, 0, Real{0.5}, -1, 0, Real{0.5}, 0, 1})};
}

template <typename Real>
auto uniform_scale(Real scale) -> std::array<Real, 16> {
  return {scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, 1};
}

/// Core's own polygons, read as the arrays the Python surface states.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto face_array(const tf::polygons_buffer<Index, Real, Dims, Ngon> &value)
    -> tf::cpp::nd_array<Index> {
  const auto count = static_cast<int>(value.faces_buffer().size());
  return tf::cpp::test::face_indices_of(value).reshape({count, 3});
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto point_array(const tf::polygons_buffer<Index, Real, Dims, Ngon> &value)
    -> tf::cpp::nd_array<Real> {
  const auto &coordinates = value.points_buffer().data_buffer();
  const auto count = static_cast<int>(value.points_buffer().size());
  return tf::cpp::test::copied_nd_array(coordinates,
                                        {count, static_cast<int>(Dims)});
}

template <typename RealT, typename IndexT, std::size_t CoordinateDims>
struct boundary_axis {
  using Real = RealT;
  using Index = IndexT;
  static constexpr std::size_t Dims = CoordinateDims;
};

using boundary_f32_i32_2 = boundary_axis<float, std::int32_t, 2>;
using boundary_f32_i32_3 = boundary_axis<float, std::int32_t, 3>;
using boundary_f32_i64_2 = boundary_axis<float, std::int64_t, 2>;
using boundary_f32_i64_3 = boundary_axis<float, std::int64_t, 3>;
using boundary_f64_i32_2 = boundary_axis<double, std::int32_t, 2>;
using boundary_f64_i32_3 = boundary_axis<double, std::int32_t, 3>;
using boundary_f64_i64_2 = boundary_axis<double, std::int64_t, 2>;
using boundary_f64_i64_3 = boundary_axis<double, std::int64_t, 3>;

/// The coordinates a planar fixture stands on, padded to this axis's Dims.
template <typename Axis>
auto planar_points(std::initializer_list<double> xy)
    -> std::vector<typename Axis::Real> {
  using Real = typename Axis::Real;
  const auto count = xy.size() / 2;
  std::vector<Real> coordinates(count * Axis::Dims);
  auto input = xy.begin();
  for (std::size_t point = 0; point < count; ++point) {
    coordinates[point * Axis::Dims] = static_cast<Real>(*input++);
    coordinates[point * Axis::Dims + 1] = static_cast<Real>(*input++);
  }
  return coordinates;
}

template <typename Axis>
auto planar_array(std::initializer_list<double> xy)
    -> tf::cpp::nd_array<typename Axis::Real> {
  const auto coordinates = planar_points<Axis>(xy);
  return tf::cpp::test::make_nd_array(
      coordinates,
      {static_cast<int>(xy.size() / 2), static_cast<int>(Axis::Dims)});
}

template <typename Axis, std::size_t Ngon>
using boundary_mesh =
    tf::cpp::test::owned_mesh<typename Axis::Index, typename Axis::Real,
                              Axis::Dims, Ngon>;

template <typename Axis, std::size_t Ngon>
auto boundary_fixture(std::initializer_list<typename Axis::Index> faces,
                      int face_count, std::initializer_list<double> xy)
    -> boundary_mesh<Axis, Ngon> {
  using Real = typename Axis::Real;
  using Index = typename Axis::Index;
  const auto points = planar_points<Axis>(xy);
  if constexpr (Ngon == 3) {
    static_cast<void>(face_count);
    return {tf::cpp::test::polygons_of<Index, Real, Axis::Dims>(faces, points)};
  } else {
    std::vector<Index> offsets(static_cast<std::size_t>(face_count + 1));
    for (int face = 0; face <= face_count; ++face)
      offsets[static_cast<std::size_t>(face)] = static_cast<Index>(face * 3);
    return {tf::cpp::test::polygons_of<Index, Real, Axis::Dims>(offsets, faces,
                                                                points)};
  }
}

template <typename Axis, std::size_t Ngon>
auto boundary_single_triangle() -> boundary_mesh<Axis, Ngon> {
  using Index = typename Axis::Index;
  return boundary_fixture<Axis, Ngon>({Index{0}, Index{1}, Index{2}}, 1,
                                      {0, 0, 1, 0, 0.5, 1});
}

template <typename Axis, std::size_t Ngon>
auto boundary_two_triangles() -> boundary_mesh<Axis, Ngon> {
  using Index = typename Axis::Index;
  return boundary_fixture<Axis, Ngon>(
      {Index{0}, Index{1}, Index{2}, Index{1}, Index{3}, Index{2}}, 2,
      {0, 0, 1, 0, 0.5, 1, 1.5, 1});
}

template <typename Axis, std::size_t Ngon>
auto boundary_tetrahedron() -> boundary_mesh<Axis, Ngon> {
  using Index = typename Axis::Index;
  return boundary_fixture<Axis, Ngon>({Index{0}, Index{1}, Index{2}, Index{0},
                                       Index{2}, Index{3}, Index{0}, Index{3},
                                       Index{1}, Index{1}, Index{3}, Index{2}},
                                      4, {0, 0, 1, 0, 0.5, 1, 0.5, 0.5});
}

template <typename Axis, std::size_t Ngon>
auto boundary_mesh_with_hole() -> boundary_mesh<Axis, Ngon> {
  using Index = typename Axis::Index;
  return boundary_fixture<Axis, Ngon>(
      {Index{0}, Index{1}, Index{4}, Index{1}, Index{5}, Index{4}, Index{1},
       Index{2}, Index{5}, Index{2}, Index{6}, Index{5}, Index{2}, Index{3},
       Index{6}, Index{3}, Index{4}, Index{6}, Index{3}, Index{0}, Index{4}},
      7, {0, 0, 1, 0, 1, 1, 0, 1, 0.3, 0.3, 0.7, 0.3, 0.5, 0.7});
}

template <typename Index>
auto canonicalize_edges(const tf::cpp::nd_array<Index> &edges)
    -> std::set<std::pair<Index, Index>> {
  std::set<std::pair<Index, Index>> result;
  for (int edge = 0; edge < edges.shape_at(0); ++edge) {
    const auto first = edges[static_cast<std::size_t>(edge * 2)];
    const auto second = edges[static_cast<std::size_t>(edge * 2 + 1)];
    result.emplace(std::min(first, second), std::max(first, second));
  }
  return result;
}

template <typename Axis, std::size_t Ngon>
auto check_boundary_cache_case() -> void {
  auto owned = boundary_two_triangles<Axis, Ngon>();
  REQUIRE_FALSE(owned.cache.is_face_membership_built());
  REQUIRE(owned.cache.face_membership_build_count() == 0);

  CHECK(tf::cpp::boundary_edges(owned.mesh()).shape_at(0) == 4);
  REQUIRE(owned.cache.is_face_membership_fresh(owned.mesh().geometry()));
  REQUIRE(owned.cache.face_membership_build_count() == 1);
  CHECK(tf::cpp::boundary_paths(owned.mesh()).size() == 1);
  CHECK(tf::cpp::boundary_curves(owned.mesh()).paths.size() == 1);
  CHECK(owned.cache.face_membership_build_count() == 1);

  // a caller that rewires its connectivity says so, and the statement is what
  // retires what was known of it
  owned.cache.faces_changed();
  REQUIRE_FALSE(owned.cache.is_face_membership_fresh(owned.mesh().geometry()));

  CHECK(tf::cpp::boundary_curves(owned.mesh()).paths.size() == 1);
  REQUIRE(owned.cache.is_face_membership_fresh(owned.mesh().geometry()));
  CHECK(owned.cache.face_membership_build_count() == 2);
  CHECK(tf::cpp::boundary_edges(owned.mesh()).shape_at(0) == 4);
  CHECK(tf::cpp::boundary_paths(owned.mesh()).size() == 1);
  CHECK(owned.cache.face_membership_build_count() == 2);
}

} // namespace

TEMPLATE_TEST_CASE(
    "Python parity: mesh topology links expose exact fixed-mesh structure",
    "[cpp][topology][python-parity][links][cache]", float, double) {
  auto owned = two_triangles<TestType>();
  REQUIRE_FALSE(owned.cache.is_face_membership_built());
  REQUIRE_FALSE(owned.cache.is_manifold_edge_link_built());
  REQUIRE_FALSE(owned.cache.is_face_link_built());
  REQUIRE_FALSE(owned.cache.is_vertex_link_built());

  const auto mesh = owned.mesh();
  const auto membership = owned.cache.face_membership_handle(mesh.geometry());
  CHECK(equals(membership.offsets(), {0, 1, 3, 5, 6}));
  CHECK(block_equals(membership, 0, {0}));
  CHECK(block_equals(membership, 1, {0, 1}));
  CHECK(block_equals(membership, 2, {0, 1}));
  CHECK(block_equals(membership, 3, {1}));
  CHECK(owned.cache.face_membership_build_count() == 1);

  const auto manifold = owned.cache.manifold_edge_link_handle(mesh.geometry());
  CHECK(manifold.raw_shape() == tf::small_vector<int, 3>{2, 3});
  CHECK(equals(manifold, {-1, 1, -1, -1, -1, 0}));

  const auto face_link = owned.cache.face_link_handle(mesh.geometry());
  CHECK(equals(face_link.offsets(), {0, 1, 2}));
  CHECK(block_equals(face_link, 0, {1}));
  CHECK(block_equals(face_link, 1, {0}));

  const auto vertex_link = owned.cache.vertex_link_handle(mesh.geometry());
  CHECK(equals(vertex_link.offsets(), {0, 2, 5, 8, 10}));
  CHECK(block_equals(vertex_link, 0, {1, 2}));
  CHECK(block_equals(vertex_link, 1, {0, 2, 3}));
  CHECK(block_equals(vertex_link, 2, {0, 1, 3}));
  CHECK(block_equals(vertex_link, 3, {1, 2}));
  CHECK(owned.cache.is_face_membership_fresh(mesh.geometry()));
  CHECK(owned.cache.is_manifold_edge_link_fresh(mesh.geometry()));
  CHECK(owned.cache.is_face_link_fresh(mesh.geometry()));
  CHECK(owned.cache.is_vertex_link_fresh(mesh.geometry()));

  // a placement is not what any of these are known from
  owned.place(uniform_scale<TestType>(TestType{2}));
  CHECK(owned.cache.face_membership_build_count() == 1);
  CHECK(owned.cache.manifold_edge_link_build_count() == 1);
  CHECK(owned.cache.face_link_build_count() == 1);
  CHECK(owned.cache.vertex_link_build_count() == 1);

  auto assigned_membership = membership.deep_copy();
  auto assigned_manifold = manifold.deep_copy();
  auto assigned_face_link = face_link.deep_copy();
  auto assigned_vertex_link = vertex_link.deep_copy();
  owned.cache.set_face_membership(assigned_membership, mesh.geometry());
  owned.cache.set_manifold_edge_link(assigned_manifold, mesh.geometry());
  owned.cache.set_face_link(assigned_face_link, mesh.geometry());
  owned.cache.set_vertex_link(assigned_vertex_link, mesh.geometry());
  assigned_membership.destroy();
  assigned_manifold.destroy();
  assigned_face_link.destroy();
  assigned_vertex_link.destroy();
  // what a caller states is the cache's from then on, whatever the caller
  // does with the handle it stated it through
  CHECK(equals(owned.cache.face_membership_handle(mesh.geometry()).offsets(),
               {0, 1, 3, 5, 6}));
  CHECK(equals(owned.cache.manifold_edge_link_handle(mesh.geometry()),
               {-1, 1, -1, -1, -1, 0}));
  CHECK(equals(owned.cache.face_link_handle(mesh.geometry()).offsets(),
               {0, 1, 2}));
  CHECK(equals(owned.cache.vertex_link_handle(mesh.geometry()).offsets(),
               {0, 2, 5, 8, 10}));
  CHECK(owned.cache.face_membership_build_count() == 1);

  CHECK_THROWS_AS(
      owned.cache.set_manifold_edge_link(
          make_array<std::int32_t>({-1, -1, -1}, {1, 3}), mesh.geometry()),
      std::invalid_argument);
  CHECK_THROWS_AS(
      owned.cache.set_manifold_edge_link(
          make_array<std::int32_t>({-1, -1, -1, -1}, {2, 2}), mesh.geometry()),
      std::invalid_argument);

  owned.cache.faces_changed();
  const auto restated = owned.mesh();
  CHECK_FALSE(owned.cache.is_face_membership_fresh(restated.geometry()));
  CHECK_FALSE(owned.cache.is_manifold_edge_link_fresh(restated.geometry()));
  CHECK_FALSE(owned.cache.is_face_link_fresh(restated.geometry()));
  CHECK_FALSE(owned.cache.is_vertex_link_fresh(restated.geometry()));
  CHECK(
      equals(owned.cache.face_membership_handle(restated.geometry()).offsets(),
             {0, 1, 3, 5, 6}));
  CHECK(equals(owned.cache.manifold_edge_link_handle(restated.geometry()),
               {-1, 1, -1, -1, -1, 0}));
  CHECK(equals(owned.cache.face_link_handle(restated.geometry()).offsets(),
               {0, 1, 2}));
  CHECK(equals(owned.cache.vertex_link_handle(restated.geometry()).offsets(),
               {0, 2, 5, 8, 10}));
  CHECK(owned.cache.face_membership_build_count() == 2);

  // a structure is handed back RETAINED, so it outlives the cache it came from
  auto survivor = owned.cache.vertex_link_handle(restated.geometry());
  owned = {};
  CHECK(equals(survivor.offsets(), {0, 2, 5, 8, 10}));
  CHECK(block_equals(survivor, 2, {0, 1, 3}));
}

// One name for one operation: orient_faces_consistently is the only spelling,
// at every index the matrix carries.
TEMPLATE_TEST_CASE(
    "Python parity: consistent orientation is topology-only and non-mutating",
    "[cpp][topology][python-parity][orientation][ownership]", float, double) {
  SECTION("already consistent and disconnected faces stay exact") {
    auto source = two_triangles<TestType>();
    const auto source_faces = face_array(source.polygons);
    auto result = tf::cpp::orient_faces_consistently(source.mesh());
    CHECK(same_array(face_array(result), source_faces));

    parity_mesh<TestType> disconnected{
        tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
            {0, 1, 2, 3, 4, 5},
            {0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 0, 0, 4, 0, 0, 3, 1, 0})};
    auto disconnected_result =
        tf::cpp::orient_faces_consistently(disconnected.mesh());
    CHECK(same_array(face_array(disconnected_result),
                     face_array(disconnected.polygons)));
  }

  SECTION("inconsistent components are fixed without changing identity") {
    auto source = two_triangles<TestType>(true);
    tf::cpp::build_manifold_edge_link(source.mesh());
    source.place(uniform_scale<TestType>(TestType{3}));
    const auto original_faces = face_array(source.polygons);
    const auto *source_coordinates =
        source.polygons.points_buffer().data_buffer().begin();

    auto result = tf::cpp::orient_faces_consistently(source.mesh());
    CHECK(has_consistent_shared_edges(face_array(result)));
    CHECK(same_face_vertices(face_array(result), original_faces));
    CHECK_FALSE(same_array(face_array(result), original_faces));
    CHECK(same_array(face_array(source.polygons), original_faces));
    CHECK(source.cache.is_manifold_edge_link_fresh(source.mesh().geometry()));
    CHECK(result.faces_buffer().data_buffer().begin() !=
          source.polygons.faces_buffer().data_buffer().begin());
    CHECK(result.points_buffer().data_buffer().begin() != source_coordinates);

    source = {};
    CHECK(has_consistent_shared_edges(face_array(result)));
    CHECK(result.points_buffer().data_buffer()[3] == TestType{1});
  }

  SECTION("one flipped face in a fan is repaired") {
    parity_mesh<TestType> source{
        tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
            {0, 1, 4, 1, 2, 4, 4, 3, 2, 3, 0, 4},
            {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, TestType{0.5}, TestType{0.5},
             0})};
    const auto original_faces = face_array(source.polygons);
    auto result = tf::cpp::orient_faces_consistently(source.mesh());
    CHECK(has_consistent_shared_edges(face_array(result)));
    CHECK(same_face_vertices(face_array(result), original_faces));
    CHECK(same_array(face_array(source.polygons), original_faces));
  }

  // a caller holding nothing holds the EMPTY mesh, which is already oriented
  parity_mesh<TestType> nothing;
  CHECK(tf::cpp::orient_faces_consistently(nothing.mesh())
            .faces_buffer()
            .size() == 0);
}

TEMPLATE_TEST_CASE(
    "Python parity: mesh analysis predicates cover the fixed-mesh truth table",
    "[cpp][topology][python-parity][analysis][transform]", float, double) {
  const auto single_owner = single_triangle<TestType>();
  auto open_owner = two_triangles<TestType>();
  const auto closed_owner = tetrahedron<TestType>();
  const auto non_manifold_owner = non_manifold_mesh<TestType>();
  const auto single = single_owner.mesh();
  const auto open = open_owner.mesh();
  const auto closed = closed_owner.mesh();
  const auto non_manifold = non_manifold_owner.mesh();

  CHECK_FALSE(tf::cpp::is_closed(single));
  CHECK(tf::cpp::is_open(single));
  CHECK(tf::cpp::is_manifold(single));
  CHECK_FALSE(tf::cpp::is_non_manifold(single));

  CHECK_FALSE(tf::cpp::is_closed(open));
  CHECK(tf::cpp::is_open(open));
  CHECK(tf::cpp::is_manifold(open));
  CHECK_FALSE(tf::cpp::is_non_manifold(open));

  CHECK(tf::cpp::is_closed(closed));
  CHECK_FALSE(tf::cpp::is_open(closed));
  CHECK(tf::cpp::is_manifold(closed));
  CHECK_FALSE(tf::cpp::is_non_manifold(closed));

  CHECK_FALSE(tf::cpp::is_manifold(non_manifold));
  CHECK(tf::cpp::is_non_manifold(non_manifold));
  CHECK(tf::cpp::is_closed(closed) != tf::cpp::is_open(closed));
  CHECK(tf::cpp::is_manifold(non_manifold) !=
        tf::cpp::is_non_manifold(non_manifold));

  tf::cpp::build_face_membership(open);
  open_owner.place(uniform_scale<TestType>(TestType{5}));
  CHECK_FALSE(tf::cpp::is_closed(open_owner.mesh()));
  CHECK(tf::cpp::is_open(open_owner.mesh()));
  CHECK(tf::cpp::is_manifold(open_owner.mesh()));
  CHECK(open_owner.cache.face_membership_build_count() == 1);

  // a caller holding nothing holds the EMPTY mesh: it is closed, manifold, and
  // has no boundary, because it has nothing that could break either
  parity_mesh<TestType> empty;
  const auto nothing = empty.mesh();
  CHECK(tf::cpp::is_closed(nothing));
  CHECK_FALSE(tf::cpp::is_open(nothing));
  CHECK(tf::cpp::is_manifold(nothing));
  CHECK_FALSE(tf::cpp::is_non_manifold(nothing));
}

TEMPLATE_TEST_CASE("Python parity: typed mesh boundaries match every fixed and "
                   "dynamic fixture",
                   "[cpp][topology][python-parity][boundary][typed][dynamic]",
                   boundary_f32_i32_2, boundary_f32_i32_3, boundary_f32_i64_2,
                   boundary_f32_i64_3, boundary_f64_i32_2, boundary_f64_i32_3,
                   boundary_f64_i64_2, boundary_f64_i64_3) {
  using Axis = TestType;
  using Real = typename Axis::Real;
  using Index = typename Axis::Index;
  using mesh_type = tf::cpp::mesh<Index, Real, Axis::Dims>;
  using paths_type = tf::cpp::offset_blocked_buffer<Index, Index>;
  using curves_type = tf::cpp::boundary_curves_result<Index, Real, Axis::Dims>;

  static_assert(std::is_same_v<decltype(tf::cpp::boundary_edges(
                                   std::declval<const mesh_type &>())),
                               tf::cpp::nd_array<Index>>);
  static_assert(std::is_same_v<decltype(tf::cpp::boundary_paths(
                                   std::declval<const mesh_type &>())),
                               paths_type>);
  static_assert(std::is_same_v<decltype(tf::cpp::boundary_curves(
                                   std::declval<const mesh_type &>())),
                               curves_type>);
  static_assert(
      std::is_same_v<decltype(std::declval<curves_type>().paths), paths_type>);
  static_assert(std::is_same_v<decltype(std::declval<curves_type>().points),
                               tf::cpp::nd_array<Real>>);

  SECTION("single triangle") {
    auto owned = boundary_single_triangle<Axis, 3>();
    const auto mesh = owned.mesh();
    const auto source_points = point_array(owned.polygons);
    const auto edges = tf::cpp::boundary_edges(mesh);
    CHECK(edges.raw_shape() == tf::small_vector<int, 3>{3, 2});
    CHECK(canonicalize_edges(edges) ==
          std::set<std::pair<Index, Index>>{{Index{0}, Index{1}},
                                            {Index{0}, Index{2}},
                                            {Index{1}, Index{2}}});

    const auto paths = tf::cpp::boundary_paths(mesh);
    CHECK(canonicalize_paths(paths) ==
          std::vector<std::vector<Index>>{{Index{0}, Index{1}, Index{2}}});
    REQUIRE(paths.size() == 1);
    CHECK(paths.get(0).length() == 4);
    CHECK(paths.get(0)[0] == paths.get(0)[3]);

    const auto curves = tf::cpp::boundary_curves(mesh);
    CHECK(canonicalize_paths(curves.paths) == canonicalize_paths(paths));
    CHECK(curves.points.raw_shape() ==
          tf::small_vector<int, 3>{3, static_cast<int>(Axis::Dims)});
    CHECK(same_array(curves.points, source_points));
  }

  SECTION("curves sort unique IDs and remap sparse boundary vertices") {
    auto owned = boundary_fixture<Axis, 3>({Index{4}, Index{1}, Index{3}}, 1,
                                           {10, 10, 1, 0, 20, 20, 3, 0, 4, 0});
    const auto curves = tf::cpp::boundary_curves(owned.mesh());
    CHECK(canonicalize_paths(curves.paths) ==
          std::vector<std::vector<Index>>{{Index{0}, Index{1}, Index{2}}});
    CHECK(same_array(curves.points, planar_array<Axis>({1, 0, 3, 0, 4, 0})));
  }

  SECTION("two triangles") {
    auto owned = boundary_two_triangles<Axis, 3>();
    const auto mesh = owned.mesh();
    const auto edges = tf::cpp::boundary_edges(mesh);
    CHECK(edges.raw_shape() == tf::small_vector<int, 3>{4, 2});
    CHECK(canonicalize_edges(edges) ==
          std::set<std::pair<Index, Index>>{{Index{0}, Index{1}},
                                            {Index{0}, Index{2}},
                                            {Index{1}, Index{3}},
                                            {Index{2}, Index{3}}});
    const auto paths = tf::cpp::boundary_paths(mesh);
    CHECK(canonicalize_paths(paths) ==
          std::vector<std::vector<Index>>{
              {Index{0}, Index{1}, Index{3}, Index{2}}});
    REQUIRE(paths.size() == 1);
    CHECK(paths.get(0).length() == 5);
    CHECK(paths.get(0)[0] == paths.get(0)[4]);
  }

  SECTION("closed tetrahedron has canonical empty curves") {
    auto owned = boundary_tetrahedron<Axis, 3>();
    const auto mesh = owned.mesh();
    const auto edges = tf::cpp::boundary_edges(mesh);
    const auto paths = tf::cpp::boundary_paths(mesh);
    const auto curves = tf::cpp::boundary_curves(mesh);
    CHECK(edges.raw_shape() == tf::small_vector<int, 3>{0, 2});
    CHECK(paths.size() == 0);
    CHECK(paths.data().empty());
    CHECK(curves.paths.is_valid());
    CHECK(curves.paths.size() == 0);
    CHECK(curves.paths.offsets().empty());
    CHECK(curves.paths.data().empty());
    CHECK(curves.points.raw_shape() ==
          tf::small_vector<int, 3>{0, static_cast<int>(Axis::Dims)});
  }

  SECTION("outer boundary and hole remain separate loops") {
    auto owned = boundary_mesh_with_hole<Axis, 3>();
    const auto mesh = owned.mesh();
    const auto edges = tf::cpp::boundary_edges(mesh);
    CHECK(edges.shape_at(0) == 7);
    const auto paths = tf::cpp::boundary_paths(mesh);
    CHECK(canonicalize_paths(paths) ==
          std::vector<std::vector<Index>>{
              {Index{0}, Index{1}, Index{2}, Index{3}},
              {Index{4}, Index{5}, Index{6}}});
    const auto curves = tf::cpp::boundary_curves(mesh);
    CHECK(canonicalize_paths(curves.paths) == canonicalize_paths(paths));
    CHECK(curves.points.raw_shape() ==
          tf::small_vector<int, 3>{7, static_cast<int>(Axis::Dims)});
    CHECK(same_array(curves.points, point_array(owned.polygons)));
  }

  SECTION("dynamic single triangle") {
    auto owned = boundary_single_triangle<Axis, tf::dynamic_size>();
    const auto mesh = owned.mesh();
    const auto edges = tf::cpp::boundary_edges(mesh);
    CHECK(canonicalize_edges(edges) ==
          std::set<std::pair<Index, Index>>{{Index{0}, Index{1}},
                                            {Index{0}, Index{2}},
                                            {Index{1}, Index{2}}});
    const auto paths = tf::cpp::boundary_paths(mesh);
    CHECK(canonicalize_paths(paths) ==
          std::vector<std::vector<Index>>{{Index{0}, Index{1}, Index{2}}});
    const auto curves = tf::cpp::boundary_curves(mesh);
    CHECK(canonicalize_paths(curves.paths) == canonicalize_paths(paths));
    CHECK(same_array(curves.points, point_array(owned.polygons)));
  }

  SECTION("dynamic two triangles") {
    auto owned = boundary_two_triangles<Axis, tf::dynamic_size>();
    const auto mesh = owned.mesh();
    CHECK(canonicalize_edges(tf::cpp::boundary_edges(mesh)) ==
          std::set<std::pair<Index, Index>>{{Index{0}, Index{1}},
                                            {Index{0}, Index{2}},
                                            {Index{1}, Index{3}},
                                            {Index{2}, Index{3}}});
    CHECK(canonicalize_paths(tf::cpp::boundary_paths(mesh)) ==
          std::vector<std::vector<Index>>{
              {Index{0}, Index{1}, Index{3}, Index{2}}});
  }

  SECTION("dynamic quad does not assume fixed triangle arity") {
    boundary_mesh<Axis, tf::dynamic_size> owned{
        tf::cpp::test::polygons_of<Index, Real, Axis::Dims>(
            std::vector<Index>{Index{0}, Index{4}},
            std::vector<Index>{Index{0}, Index{1}, Index{2}, Index{3}},
            planar_points<Axis>({0, 0, 1, 0, 1, 1, 0, 1}))};
    const auto mesh = owned.mesh();
    CHECK(tf::cpp::boundary_edges(mesh).shape_at(0) == 4);
    CHECK(canonicalize_paths(tf::cpp::boundary_paths(mesh)) ==
          std::vector<std::vector<Index>>{
              {Index{0}, Index{1}, Index{2}, Index{3}}});
  }
}

TEMPLATE_TEST_CASE(
    "Python parity: typed mesh boundaries reject malformed connectivity safely",
    "[cpp][topology][python-parity][boundary][validation]", boundary_f32_i32_2,
    boundary_f32_i32_3, boundary_f32_i64_2, boundary_f32_i64_3,
    boundary_f64_i32_2, boundary_f64_i32_3, boundary_f64_i64_2,
    boundary_f64_i64_3) {
  using Axis = TestType;
  using Index = typename Axis::Index;
  using Real = typename Axis::Real;

  // the library cannot see a caller's storage, so a face index is refused
  // where the READING is read, at either arity
  auto fixed = boundary_single_triangle<Axis, 3>();
  tf::cpp::test::fill_storage(fixed.polygons.points_buffer().data_buffer(),
                              planar_points<Axis>({0, 0, 1, 0}));
  fixed.cache.points_changed();
  CHECK_THROWS_AS(tf::cpp::boundary_edges(fixed.mesh()), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::boundary_paths(fixed.mesh()), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::boundary_curves(fixed.mesh()), std::out_of_range);

  boundary_mesh<Axis, tf::dynamic_size> unreachable{
      tf::cpp::test::polygons_of<Index, Real, Axis::Dims>(
          std::vector<Index>{Index{0}, Index{3}},
          std::vector<Index>{Index{0}, Index{1}, Index{3}},
          planar_points<Axis>({0, 0, 1, 0, 0, 1}))};
  CHECK_THROWS_AS(tf::cpp::boundary_edges(unreachable.mesh()),
                  std::out_of_range);

  auto invalid_index = boundary_single_triangle<Axis, tf::dynamic_size>();
  tf::cpp::test::fill_storage(
      invalid_index.polygons.points_buffer().data_buffer(),
      planar_points<Axis>({0, 0, 1, 0}));
  invalid_index.cache.points_changed();
  CHECK_THROWS_AS(tf::cpp::boundary_edges(invalid_index.mesh()),
                  std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::boundary_paths(invalid_index.mesh()),
                  std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::boundary_curves(invalid_index.mesh()),
                  std::out_of_range);
}

TEST_CASE("boundary operations warm and reuse authoritative membership cache",
          "[cpp][topology][python-parity][boundary][cache][freshness]") {
  using Axis = boundary_f64_i64_3;

  check_boundary_cache_case<Axis, 3>();
  check_boundary_cache_case<Axis, tf::dynamic_size>();
}

TEMPLATE_TEST_CASE(
    "Python parity: typed edges connect to paths across every Python case",
    "[cpp][topology][python-parity][paths][validation]", std::int32_t,
    std::int64_t) {
  using Index = TestType;
  using path_buffer = tf::cpp::offset_blocked_buffer<Index, Index>;

  static_assert(
      supports_connect_edges_to_paths<tf::cpp::nd_array<Index>>::value);
  static_assert(
      !supports_connect_edges_to_paths<tf::cpp::nd_array<float>>::value);

  SECTION("simple line") {
    const auto paths = tf::cpp::connect_edges_to_paths(
        make_array<Index>({0, 1, 1, 2, 2, 3}, {3, 2}));
    static_assert(std::is_same_v<std::decay_t<decltype(paths)>, path_buffer>);
    CHECK(paths.size() == 1);
    CHECK(canonicalize_paths(paths) ==
          std::vector<std::vector<Index>>{{0, 1, 2, 3}});
  }

  SECTION("closed loop") {
    const auto paths = tf::cpp::connect_edges_to_paths(
        make_array<Index>({0, 1, 1, 2, 2, 3, 3, 0}, {4, 2}));
    REQUIRE(paths.size() == 1);
    const auto path = paths.get(0);
    REQUIRE(path.length() > 1);
    CHECK(path[0] == path[path.length() - 1]);
    CHECK(canonicalize_paths(paths) ==
          std::vector<std::vector<Index>>{{0, 1, 2, 3}});
  }

  SECTION("two separate lines") {
    const auto paths = tf::cpp::connect_edges_to_paths(
        make_array<Index>({0, 1, 1, 2, 3, 4, 4, 5}, {4, 2}));
    CHECK(paths.size() == 2);
    CHECK(canonicalize_paths(paths) ==
          std::vector<std::vector<Index>>{{0, 1, 2}, {3, 4, 5}});
  }

  SECTION("unordered edges") {
    const auto paths = tf::cpp::connect_edges_to_paths(
        make_array<Index>({2, 3, 0, 1, 1, 2}, {3, 2}));
    CHECK(paths.size() == 1);
    CHECK(canonicalize_paths(paths) ==
          std::vector<std::vector<Index>>{{0, 1, 2, 3}});
  }

  SECTION("single edge") {
    const auto paths =
        tf::cpp::connect_edges_to_paths(make_array<Index>({0, 1}, {1, 2}));
    REQUIRE(paths.size() == 1);
    CHECK(paths.get(0).length() == 2);
    CHECK(canonicalize_paths(paths) == std::vector<std::vector<Index>>{{0, 1}});
  }

  SECTION("triangle") {
    const auto paths = tf::cpp::connect_edges_to_paths(
        make_array<Index>({0, 1, 1, 2, 2, 0}, {3, 2}));
    REQUIRE(paths.size() == 1);
    const auto path = paths.get(0);
    REQUIRE(path.length() > 1);
    CHECK(path[0] == path[path.length() - 1]);
    CHECK(canonicalize_paths(paths) ==
          std::vector<std::vector<Index>>{{0, 1, 2}});
  }

  SECTION("empty input") {
    const auto paths =
        tf::cpp::connect_edges_to_paths(make_array<Index>({}, {0, 2}));
    CHECK(paths.is_valid());
    CHECK(paths.size() == 0);
    CHECK(paths.data().empty());
    CHECK(paths.offsets().empty());
  }

  SECTION("branching structure") {
    const auto paths = tf::cpp::connect_edges_to_paths(
        make_array<Index>({0, 1, 1, 2, 1, 3}, {3, 2}));
    REQUIRE(paths.size() > 0);
    const auto data = paths.data();
    const std::set<Index> vertices(data.begin(), data.end());
    CHECK(vertices == std::set<Index>{0, 1, 2, 3});
  }

  SECTION("figure eight") {
    const auto paths = tf::cpp::connect_edges_to_paths(
        make_array<Index>({0, 1, 1, 2, 2, 0, 2, 3, 3, 4, 4, 2}, {6, 2}));
    REQUIRE(paths.size() > 0);
    const auto data = paths.data();
    const std::set<Index> vertices(data.begin(), data.end());
    CHECK(vertices == std::set<Index>{0, 1, 2, 3, 4});
  }

  SECTION("dtype preservation") {
    const auto paths = tf::cpp::connect_edges_to_paths(
        make_array<Index>({0, 1, 1, 2}, {2, 2}));
    static_assert(
        std::is_same_v<decltype(paths.offsets()), tf::cpp::nd_array<Index>>);
    static_assert(
        std::is_same_v<decltype(paths.data()), tf::cpp::nd_array<Index>>);
  }

  SECTION("valid offset-block structure") {
    const auto paths = tf::cpp::connect_edges_to_paths(
        make_array<Index>({0, 1, 1, 2, 3, 4}, {3, 2}));
    CHECK(has_valid_offsets(paths));
  }

  SECTION("malformed shapes") {
    CHECK_THROWS_AS(
        tf::cpp::connect_edges_to_paths(make_array<Index>({0, 1, 2}, {3})),
        std::invalid_argument);
    CHECK_THROWS_AS(
        tf::cpp::connect_edges_to_paths(make_array<Index>({0, 1, 2}, {1, 3})),
        std::invalid_argument);
  }
}

TEMPLATE_TEST_CASE(
    "Python parity: connected component labels are exact across carriers",
    "[cpp][topology][python-parity][components][ownership]", std::int32_t,
    std::int64_t) {
  using Index = TestType;
  using expected_result = tf::cpp::connected_components_result<Index>;

  static_assert(
      !supports_connected_component_labels<tf::cpp::nd_array<float>>::value);
  static_assert(!supports_connected_component_labels<
                tf::cpp::offset_blocked_buffer<float, Index>>::value);
  static_assert(!supports_connected_component_labels<
                tf::cpp::offset_blocked_buffer<Index, float>>::value);

  auto fixed = make_array<Index>(
      {1, -1, -1, 0, 2, -1, 1, -1, -1, 4, -1, -1, 3, 5, -1, 4, -1, -1}, {6, 3});
  auto fixed_result = tf::cpp::label_connected_components(fixed);
  static_assert(std::is_same_v<decltype(fixed_result), expected_result>);
  static_assert(
      std::is_same_v<decltype(fixed_result.labels), tf::cpp::nd_array<Index>>);
  static_assert(std::is_same_v<decltype(fixed_result.n_components), Index>);
  CHECK(fixed_result.n_components == Index{2});
  CHECK(equals(fixed_result.labels, {0, 0, 0, 1, 1, 1}));

  auto variable = tf::cpp::offset_blocked_buffer<Index, Index>::create(
      make_array<Index>({0, 1, 3, 4, 5, 7, 8}, {7}),
      make_array<Index>({1, 0, 2, 1, 4, 3, 5, 4}, {8}));
  auto variable_result = tf::cpp::label_connected_components(variable);
  CHECK(variable_result.n_components == Index{2});
  CHECK(same_array(variable_result.labels, fixed_result.labels));

  const auto fixed_small_hint = tf::cpp::label_connected_components(fixed, 2);
  const auto fixed_large_hint =
      tf::cpp::label_connected_components(fixed, 1000);
  const auto variable_small_hint =
      tf::cpp::label_connected_components(variable, 2);
  const auto variable_large_hint =
      tf::cpp::label_connected_components(variable, 1000);
  CHECK(fixed_small_hint.n_components == fixed_result.n_components);
  CHECK(fixed_large_hint.n_components == fixed_result.n_components);
  CHECK(variable_small_hint.n_components == variable_result.n_components);
  CHECK(variable_large_hint.n_components == variable_result.n_components);
  CHECK(same_array(fixed_small_hint.labels, fixed_result.labels));
  CHECK(same_array(fixed_large_hint.labels, fixed_result.labels));
  CHECK(same_array(variable_small_hint.labels, variable_result.labels));
  CHECK(same_array(variable_large_hint.labels, variable_result.labels));
  if constexpr (std::is_same_v<Index, std::int64_t>) {
    const auto wide_hint = tf::cpp::label_connected_components(
        fixed, std::numeric_limits<std::int64_t>::max());
    CHECK(wide_hint.n_components == fixed_result.n_components);
    CHECK(same_array(wide_hint.labels, fixed_result.labels));
  }

  auto isolated =
      make_array<Index>({-1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, {5, 2});
  auto isolated_result = tf::cpp::label_connected_components(isolated);
  auto isolated_variable = tf::cpp::offset_blocked_buffer<Index, Index>::create(
      make_array<Index>({0, 0, 0, 0, 0, 0}, {6}), make_array<Index>({}, {0}));
  auto isolated_variable_result =
      tf::cpp::label_connected_components(isolated_variable);
  CHECK(isolated_result.n_components == Index{5});
  CHECK(isolated_variable_result.n_components == Index{5});
  CHECK(equals(isolated_result.labels, {0, 1, 2, 3, 4}));
  CHECK(same_array(isolated_variable_result.labels, isolated_result.labels));

  auto fully_connected =
      make_array<Index>({1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2}, {4, 3});
  auto fully_connected_variable =
      tf::cpp::offset_blocked_buffer<Index, Index>::create(
          make_array<Index>({0, 3, 6, 9, 12}, {5}),
          make_array<Index>({1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2}, {12}));
  auto fully_connected_result =
      tf::cpp::label_connected_components(fully_connected);
  auto fully_connected_variable_result =
      tf::cpp::label_connected_components(fully_connected_variable);
  CHECK(fully_connected_result.n_components == Index{1});
  CHECK(fully_connected_variable_result.n_components == Index{1});
  CHECK(equals(fully_connected_result.labels, {0, 0, 0, 0}));
  CHECK(same_array(fully_connected_variable_result.labels,
                   fully_connected_result.labels));

  auto singleton_dense = make_array<Index>({-1, -1}, {1, 2});
  auto singleton_variable =
      tf::cpp::offset_blocked_buffer<Index, Index>::create(
          make_array<Index>({0, 0}, {2}), make_array<Index>({}, {0}));
  CHECK(
      equals(tf::cpp::label_connected_components(singleton_dense).labels, {0}));
  CHECK(tf::cpp::label_connected_components(singleton_dense).n_components ==
        Index{1});
  CHECK(equals(tf::cpp::label_connected_components(singleton_variable).labels,
               {0}));
  CHECK(tf::cpp::label_connected_components(singleton_variable).n_components ==
        Index{1});

  auto empty_dense = make_array<Index>({}, {0, 2});
  auto empty_variable = tf::cpp::offset_blocked_buffer<Index, Index>::create(
      make_array<Index>({0}, {1}), make_array<Index>({}, {0}));
  const auto empty_dense_result =
      tf::cpp::label_connected_components(empty_dense);
  const auto empty_variable_result =
      tf::cpp::label_connected_components(empty_variable);
  CHECK(empty_dense_result.n_components == Index{0});
  CHECK(empty_dense_result.labels.raw_shape() == tf::small_vector<int, 3>{0});
  CHECK(empty_variable_result.n_components == Index{0});
  CHECK(empty_variable_result.labels.raw_shape() ==
        tf::small_vector<int, 3>{0});

  variable.destroy();
  fixed.destroy();
  CHECK(equals(variable_result.labels, {0, 0, 0, 1, 1, 1}));

  tf::cpp::offset_blocked_buffer<Index, Index> invalid;
  CHECK_THROWS_AS(tf::cpp::label_connected_components(invalid),
                  std::invalid_argument);
  auto one_dimensional = make_array<Index>({0, 1}, {2});
  CHECK_THROWS_AS(tf::cpp::label_connected_components(one_dimensional),
                  std::invalid_argument);

  auto invalid_dense_peer =
      make_array<Index>({std::numeric_limits<Index>::max()}, {1, 1});
  CHECK_THROWS_AS(tf::cpp::label_connected_components(invalid_dense_peer),
                  std::out_of_range);
  auto invalid_dense_sentinel = make_array<Index>({Index{-2}}, {1, 1});
  CHECK_THROWS_AS(tf::cpp::label_connected_components(invalid_dense_sentinel),
                  std::out_of_range);
  auto invalid_variable_peer =
      tf::cpp::offset_blocked_buffer<Index, Index>::create(
          make_array<Index>({0, 1}, {2}),
          make_array<Index>({std::numeric_limits<Index>::max()}, {1}));
  CHECK_THROWS_AS(tf::cpp::label_connected_components(invalid_variable_peer),
                  std::out_of_range);
  auto invalid_variable_sentinel =
      tf::cpp::offset_blocked_buffer<Index, Index>::create(
          make_array<Index>({0, 1}, {2}), make_array<Index>({-1}, {1}));
  CHECK_THROWS_AS(
      tf::cpp::label_connected_components(invalid_variable_sentinel),
      std::out_of_range);

  CHECK_THROWS_AS(tf::cpp::label_connected_components(isolated, 0),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::label_connected_components(isolated, -1),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::label_connected_components(singleton_variable, 0),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::label_connected_components(singleton_variable, -1),
                  std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "Python parity: topological and metric neighborhoods preserve blocks",
    "[cpp][topology][python-parity][neighborhoods][transform]", float, double) {
  auto owned = two_triangles<TestType>();
  const auto mesh = owned.mesh();

  const auto k1 = tf::cpp::k_rings(mesh, 1);
  CHECK(equals(k1.offsets(), {0, 2, 5, 8, 10}));
  CHECK(block_equals(k1, 0, {1, 2}));
  CHECK(block_equals(k1, 1, {0, 2, 3}));
  CHECK(block_equals(k1, 2, {0, 1, 3}));
  CHECK(block_equals(k1, 3, {1, 2}));

  const auto k2 = tf::cpp::k_rings(mesh, 2);
  CHECK(equals(k2.offsets(), {0, 3, 6, 9, 12}));
  for (int vertex = 0; vertex < 4; ++vertex) {
    const auto k1_block = k1.get(vertex);
    const auto k2_block = k2.get(vertex);
    CHECK(k2_block.length() == 3);
    for (const auto neighbor : k1_block)
      CHECK(std::find(k2_block.begin(), k2_block.end(), neighbor) !=
            k2_block.end());
  }

  const auto inclusive = tf::cpp::k_rings(mesh, 1, true);
  CHECK(equals(inclusive.offsets(), {0, 3, 7, 11, 14}));
  for (int vertex = 0; vertex < 4; ++vertex) {
    CHECK(block_equals(
        inclusive, vertex,
        vertex == 0   ? std::initializer_list<std::int32_t>{0, 1, 2}
        : vertex == 1 ? std::initializer_list<std::int32_t>{0, 1, 2, 3}
        : vertex == 2 ? std::initializer_list<std::int32_t>{0, 1, 2, 3}
                      : std::initializer_list<std::int32_t>{1, 2, 3}));
  }

  const auto metric = tf::cpp::neighborhoods(mesh, TestType{1.5});
  CHECK(same_array(metric.offsets(), k1.offsets()));
  for (int vertex = 0; vertex < 4; ++vertex)
    CHECK(block_equals(
        metric, vertex,
        vertex == 0   ? std::initializer_list<std::int32_t>{1, 2}
        : vertex == 1 ? std::initializer_list<std::int32_t>{0, 2, 3}
        : vertex == 2 ? std::initializer_list<std::int32_t>{0, 1, 3}
                      : std::initializer_list<std::int32_t>{1, 2}));

  tf::cpp::build_vertex_link(mesh);
  owned.place(uniform_scale<TestType>(TestType{2}));
  const auto scaled = tf::cpp::neighborhoods(owned.mesh(), TestType{1.5});
  CHECK(equals(scaled.offsets(), {0, 0, 0, 0, 0}));
  CHECK(scaled.data().length() == 0);
  CHECK(owned.cache.vertex_link_build_count() == 1);

  CHECK_THROWS_AS(tf::cpp::k_rings(mesh, 0), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::k_rings(mesh, -1), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::neighborhoods(mesh, TestType{0}),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::neighborhoods(mesh, std::numeric_limits<TestType>::quiet_NaN()),
      std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "Python parity: CDT maps constrained deterministic output to inputs",
    "[cpp][topology][python-parity][cdt][maps][ownership]", float, double) {
  auto points = make_array<TestType>(
      {TestType{0.5}, TestType{0.1}, TestType{0.9}, TestType{0.5},
       TestType{0.5}, TestType{0.9}, TestType{0.1}, TestType{0.5}, 0, 0, 1, 0,
       1, 1, 0, 1},
      {8, 2});
  auto edges = make_array<std::int32_t>({0, 1, 1, 2, 2, 3, 3, 0}, {4, 2});

  const auto hull = tf::cpp::make_cdt(points);
  const auto constrained = tf::cpp::make_cdt(points, edges);
  auto mapped = tf::cpp::make_cdt_with_maps(points, edges);

  REQUIRE(hull.faces.shape_at(1) == 3);
  REQUIRE(constrained.faces.shape_at(1) == 3);
  CHECK(constrained.faces.shape_at(0) < hull.faces.shape_at(0));
  CHECK(same_array(mapped.faces, constrained.faces));
  CHECK(same_array(mapped.points, constrained.points));
  CHECK(mapped.point_index_map.f.raw_shape() ==
        tf::small_vector<int, 3>{points.shape_at(0)});
  CHECK(mapped.point_index_map.kept_ids.raw_shape() ==
        tf::small_vector<int, 3>{mapped.points.shape_at(0)});

  for (const auto point : mapped.faces) {
    CHECK(point >= 0);
    CHECK(point < mapped.points.shape_at(0));
  }
  for (std::int32_t new_id = 0;
       new_id < mapped.point_index_map.kept_ids.shape_at(0); ++new_id) {
    const auto old_id =
        mapped.point_index_map.kept_ids[static_cast<std::size_t>(new_id)];
    REQUIRE(old_id >= 0);
    REQUIRE(old_id < points.shape_at(0));
    CHECK(mapped.point_index_map.f[static_cast<std::size_t>(old_id)] == new_id);
    CHECK(mapped.points[static_cast<std::size_t>(new_id * 2)] ==
          Catch::Approx(points[static_cast<std::size_t>(old_id * 2)])
              .margin(1e-6));
    CHECK(mapped.points[static_cast<std::size_t>(new_id * 2 + 1)] ==
          Catch::Approx(points[static_cast<std::size_t>(old_id * 2 + 1)])
              .margin(1e-6));
  }

  const auto *input_storage = points.raw_owner().get();
  points.destroy();
  edges.destroy();
  CHECK(mapped.points.is_valid());
  CHECK(mapped.points.raw_owner().get() != input_storage);
  CHECK(mapped.faces.shape_at(0) > 0);

  auto points_3d = make_array<TestType>({0, 0, 0, 1, 0, 0, 0, 1, 0}, {3, 3});
  CHECK_THROWS_AS(tf::cpp::make_cdt(points_3d), std::invalid_argument);
  tf::cpp::nd_array<TestType> invalid_points;
  CHECK_THROWS_AS(tf::cpp::make_cdt(invalid_points), std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "Python parity: domain labels distinguish bounded and outer sides",
    "[cpp][topology][python-parity][domains][ownership]", float, double) {
  parity_mesh<TestType> box{
      tf::cpp::make_box_mesh(TestType{1}, TestType{1}, TestType{1})};
  const auto included = tf::cpp::make_domain_labels(box.mesh());
  const auto excluded = tf::cpp::make_domain_labels(
      box.mesh(), tf::domain_config::exclude_outer_shell);

  REQUIRE(included.number_of_domains() == 2);
  REQUIRE(included.has_outer_shell_domain());
  REQUIRE(excluded.number_of_domains() == 1);
  REQUIRE_FALSE(excluded.has_outer_shell_domain());
  CHECK(excluded.outer_shell_label() == excluded.sentinel_label());
  for (std::int32_t face = 0; face < included.number_of_faces(); ++face) {
    const auto included_a = included.label(face, 0);
    const auto included_b = included.label(face, 1);
    CHECK(included_a != included_b);
    CHECK((included_a == included.outer_shell_label() ||
           included_b == included.outer_shell_label()));
    CHECK(included_a >= included.valid_label_begin());
    CHECK(included_a < included.valid_label_end());
    CHECK(included_b >= included.valid_label_begin());
    CHECK(included_b < included.valid_label_end());

    const auto excluded_a = excluded.label(face, 0);
    const auto excluded_b = excluded.label(face, 1);
    CHECK(excluded_a != excluded_b);
    CHECK((excluded_a == excluded.sentinel_label() ||
           excluded_b == excluded.sentinel_label()));
    CHECK((excluded_a == 0 || excluded_b == 0));
  }

  box.place(uniform_scale<TestType>(TestType{4}));
  const auto transformed = tf::cpp::make_domain_labels(box.mesh());
  CHECK(same_array(transformed.labels(), included.labels()));

  // the result owns its labels, so it outlives the geometry it was read from
  box = {};
  CHECK(included.labels().is_valid());
  CHECK(included.number_of_faces() == 12);
  CHECK_THROWS_AS(included.label(-1, 0), std::out_of_range);
  CHECK_THROWS_AS(included.label(0, 2), std::out_of_range);

  parity_mesh<TestType> nothing;
  CHECK(tf::cpp::make_domain_labels(nothing.mesh()).empty());
}

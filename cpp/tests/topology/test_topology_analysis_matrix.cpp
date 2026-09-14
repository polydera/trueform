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
#include "trueform/cpp/topology.hpp"

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/mesh.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
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
  std::copy(values.begin(), values.end(), buffer.begin());
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename Index>
auto make_blocks(std::initializer_list<Index> offsets,
                 std::initializer_list<Index> data)
    -> tf::cpp::offset_blocked_buffer<Index, Index> {
  return tf::cpp::offset_blocked_buffer<Index, Index>::create(
      make_array<Index>(offsets, {static_cast<int>(offsets.size())}),
      make_array<Index>(data, {static_cast<int>(data.size())}));
}

template <typename Index, typename Real, std::size_t CoordinateDims>
struct matrix_row {
  using real_type = Real;
  using index_type = Index;
  using owned_type = tf::cpp::test::owned_mesh<Index, Real, CoordinateDims>;
  static constexpr std::size_t dims = CoordinateDims;
};

using f32_i32_2 = matrix_row<std::int32_t, float, 2>;
using f32_i32_3 = matrix_row<std::int32_t, float, 3>;
using f32_i64_2 = matrix_row<std::int64_t, float, 2>;
using f32_i64_3 = matrix_row<std::int64_t, float, 3>;
using f64_i32_2 = matrix_row<std::int32_t, double, 2>;
using f64_i32_3 = matrix_row<std::int32_t, double, 3>;
using f64_i64_2 = matrix_row<std::int64_t, double, 2>;
using f64_i64_3 = matrix_row<std::int64_t, double, 3>;

template <typename Row>
auto points(std::initializer_list<double> xyz)
    -> std::vector<typename Row::real_type> {
  using Real = typename Row::real_type;
  const auto count = xyz.size() / 3;
  std::vector<Real> output(count * Row::dims);
  auto input = xyz.begin();
  for (std::size_t point = 0; point < count; ++point) {
    output[point * Row::dims] = static_cast<Real>(*input++);
    output[point * Row::dims + 1] = static_cast<Real>(*input++);
    const auto z = *input++;
    if constexpr (Row::dims == 3)
      output[point * Row::dims + 2] = static_cast<Real>(z);
  }
  return output;
}

/// Triangles, stated as this arity states them: a fixed block or offsets that
/// are i * 3 by construction.
template <typename Row, std::size_t Ngon>
auto make_mesh(std::initializer_list<typename Row::index_type> indices,
               int face_count,
               const std::vector<typename Row::real_type> &coordinates)
    -> tf::cpp::test::mesh_at<typename Row::owned_type, Ngon> {
  using Index = typename Row::index_type;
  using Real = typename Row::real_type;
  if constexpr (Ngon == 3) {
    static_cast<void>(face_count);
    return {tf::cpp::test::polygons_of<Index, Real, Row::dims>(indices,
                                                               coordinates)};
  } else {
    std::vector<Index> offsets(static_cast<std::size_t>(face_count + 1));
    for (int face = 0; face <= face_count; ++face)
      offsets[static_cast<std::size_t>(face)] = static_cast<Index>(face * 3);
    return {tf::cpp::test::polygons_of<Index, Real, Row::dims>(offsets, indices,
                                                               coordinates)};
  }
}

template <typename Row, std::size_t Ngon>
auto two_triangles(bool inconsistent = false)
    -> tf::cpp::test::mesh_at<typename Row::owned_type, Ngon> {
  using Index = typename Row::index_type;
  return make_mesh<Row, Ngon>(
      inconsistent ? std::initializer_list<Index>{0, 1, 2, 1, 2, 3}
                   : std::initializer_list<Index>{0, 1, 2, 1, 3, 2},
      2, points<Row>({0, 0, 0, 1, 0, 0, 0.5, 1, 0, 1.5, 1, 0}));
}

template <typename Row, std::size_t Ngon>
auto tetrahedron() -> tf::cpp::test::mesh_at<typename Row::owned_type, Ngon> {
  using Index = typename Row::index_type;
  return make_mesh<Row, Ngon>(
      {Index{0}, Index{1}, Index{2}, Index{0}, Index{2}, Index{3}, Index{0},
       Index{3}, Index{1}, Index{1}, Index{3}, Index{2}},
      4, points<Row>({0, 0, 0, 1, 0, 0, 0.5, 1, 0, 0.5, 0.5, 1}));
}

template <typename Row, std::size_t Ngon>
auto non_manifold() -> tf::cpp::test::mesh_at<typename Row::owned_type, Ngon> {
  using Index = typename Row::index_type;
  return make_mesh<Row, Ngon>(
      {Index{0}, Index{1}, Index{2}, Index{0}, Index{1}, Index{3}, Index{0},
       Index{1}, Index{4}},
      3, points<Row>({0, 0, 0, 1, 0, 0, 0.5, 1, 0, 0.5, -1, 0, 0.5, 0, 1}));
}

template <typename T>
auto equals(const tf::cpp::nd_array<T> &actual,
            std::initializer_list<T> expected) -> bool {
  return actual.length() == expected.size() &&
         std::equal(actual.begin(), actual.end(), expected.begin());
}

template <typename Index>
auto equals(const tf::cpp::offset_blocked_buffer<Index, Index> &actual,
            std::initializer_list<Index> offsets,
            std::initializer_list<Index> data) -> bool {
  return equals(actual.offsets(), offsets) && equals(actual.data(), data);
}

template <typename Index>
auto equals(const tf::cpp::offset_blocked_buffer<Index, Index> &lhs,
            const tf::cpp::offset_blocked_buffer<Index, Index> &rhs) -> bool {
  return lhs.offsets().raw_shape() == rhs.offsets().raw_shape() &&
         lhs.data().raw_shape() == rhs.data().raw_shape() &&
         std::equal(lhs.offsets().begin(), lhs.offsets().end(),
                    rhs.offsets().begin()) &&
         std::equal(lhs.data().begin(), lhs.data().end(), rhs.data().begin());
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto has_consistent_shared_edges(
    const tf::polygons_buffer<Index, Real, Dims, Ngon> &value) -> bool {
  std::map<std::pair<Index, Index>, std::pair<int, int>> directions;
  for (const auto face : value.faces()) {
    auto previous = face.size() - 1;
    for (std::size_t current = 0; current < face.size(); previous = current++) {
      const auto first = face[previous];
      const auto second = face[current];
      auto &entry = directions[std::minmax(first, second)];
      ++entry.first;
      entry.second += first < second ? 1 : -1;
    }
  }
  for (const auto &entry : directions)
    if (entry.second.first == 2 && entry.second.second != 0)
      return false;
  return true;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto copied_indices(const tf::polygons_buffer<Index, Real, Dims, Ngon> &value)
    -> std::vector<Index> {
  const auto indices = tf::cpp::test::face_indices_of(value);
  return std::vector<Index>(indices.begin(), indices.end());
}

/// The faces a raw entry takes, read off the storage a test holds: rows at the
/// fixed arity, offset blocks at the mixed one.
template <typename Owned>
auto face_rows(const Owned &owned)
    -> tf::cpp::nd_array<typename Owned::index_type> {
  const auto count = static_cast<int>(owned.polygons.faces_buffer().size());
  return tf::cpp::test::face_indices_of(owned.polygons).reshape({count, 3});
}

template <typename Owned>
auto face_blocks(const Owned &owned)
    -> tf::cpp::offset_blocked_buffer<typename Owned::index_type,
                                      typename Owned::index_type> {
  using Index = typename Owned::index_type;
  return tf::cpp::offset_blocked_buffer<Index, Index>::create(
      tf::cpp::test::face_offsets_of(owned.polygons),
      tf::cpp::test::face_indices_of(owned.polygons));
}

struct mutating_resolver {
  std::shared_ptr<std::atomic<int>> submissions;
  std::function<void()> mutation;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    mutation();
    return std::make_shared<state_type<T>>();
  }
};

template <typename Row, std::size_t Ngon>
auto check_manifold_oracles_case() -> void {
  using Index = typename Row::index_type;
  auto open = two_triangles<Row, Ngon>();
  auto closed = tetrahedron<Row, Ngon>();
  auto bad = non_manifold<Row, Ngon>();

  CHECK_FALSE(tf::cpp::is_closed(open.mesh()));
  CHECK(tf::cpp::is_open(open.mesh()));
  CHECK(tf::cpp::is_manifold(open.mesh()));
  CHECK_FALSE(tf::cpp::is_non_manifold(open.mesh()));
  CHECK(tf::cpp::is_closed(closed.mesh()));
  CHECK_FALSE(tf::cpp::is_open(closed.mesh()));
  CHECK_FALSE(tf::cpp::is_manifold(bad.mesh()));
  CHECK(tf::cpp::is_non_manifold(bad.mesh()));

  const auto edges = tf::cpp::non_manifold_edges(bad.mesh());
  CHECK(edges.raw_shape() == tf::small_vector<int, 3>{1, 2});
  REQUIRE(edges.length() == 2);
  CHECK(std::min(edges[0], edges[1]) == Index{0});
  CHECK(std::max(edges[0], edges[1]) == Index{1});
  CHECK(tf::cpp::non_manifold_edges(open.mesh()).raw_shape() ==
        tf::small_vector<int, 3>{0, 2});

  // All operations consume one authoritative source membership generation.
  CHECK(open.cache.face_membership_build_count() == 1);
  CHECK(open.cache.is_face_membership_fresh(open.mesh().geometry()));
  static_cast<void>(tf::cpp::is_closed(open.mesh()));
  static_cast<void>(tf::cpp::non_manifold_edges(open.mesh()));
  CHECK(open.cache.face_membership_build_count() == 1);

  // AN ENTRY READS THE CACHE'S STRUCTURES, not fresh ones of its own: an
  // untagged form makes core build what it needs and throw it away, so what
  // states the difference is the build count. Euler stands on the edge link,
  // which nothing has asked for yet.
  CHECK(open.cache.manifold_edge_link_build_count() == 0);
  CHECK(tf::cpp::euler_characteristic(open.mesh()) == 1);
  CHECK(open.cache.manifold_edge_link_build_count() == 1);
  CHECK(open.cache.face_membership_build_count() == 1);
  CHECK(tf::cpp::euler_characteristic(open.mesh()) == 1);
  CHECK(open.cache.manifold_edge_link_build_count() == 1);

  CHECK(closed.cache.manifold_edge_link_build_count() == 0);
  CHECK(tf::cpp::euler_characteristic(closed.mesh()) == 2);
  CHECK(closed.cache.manifold_edge_link_build_count() == 1);

  // a caller that rewires its connectivity says so, and the statement is what
  // retires what was known of it
  open.cache.faces_changed();
  CHECK_FALSE(open.cache.is_face_membership_fresh(open.mesh().geometry()));
  CHECK_FALSE(tf::cpp::is_closed(open.mesh()));
  CHECK(open.cache.is_face_membership_fresh(open.mesh().geometry()));
  CHECK(open.cache.face_membership_build_count() == 2);
}

template <typename Row, std::size_t Ngon>
auto check_orient_faces_case() -> void {
  using Index = typename Row::index_type;
  auto input = two_triangles<Row, Ngon>(true);
  const auto before = copied_indices(input.polygons);
  const auto *source_coordinates =
      input.polygons.points_buffer().data_buffer().begin();
  REQUIRE_FALSE(has_consistent_shared_edges(input.polygons));

  auto output = tf::cpp::orient_faces_consistently(input.mesh());
  CHECK(has_consistent_shared_edges(output));
  CHECK(copied_indices(input.polygons) == before);
  CHECK(copied_indices(output) != before);
  // the result is storage of its own, coordinates included
  CHECK(output.points_buffer().data_buffer().begin() != source_coordinates);
  if constexpr (Ngon != 3)
    CHECK(tf::cpp::test::face_offsets_of(output).raw_shape() ==
          tf::small_vector<int, 3>{3});
  // Both arities have an edge link, and both read the source mesh's own
  // rather than building one nobody keeps.
  CHECK(input.cache.is_manifold_edge_link_fresh(input.mesh().geometry()));
  CHECK(input.cache.is_face_membership_fresh(input.mesh().geometry()));

  auto consistent = two_triangles<Row, Ngon>();
  const auto consistent_before = copied_indices(consistent.polygons);
  const auto consistent_output =
      tf::cpp::orient_faces_consistently(consistent.mesh());
  CHECK(copied_indices(consistent_output) == consistent_before);
  CHECK(has_consistent_shared_edges(consistent_output));

  auto single =
      make_mesh<Row, Ngon>({Index{0}, Index{1}, Index{2}}, 1,
                           points<Row>({0, 0, 0, 1, 0, 0, 0.5, 1, 0}));
  const auto single_before = copied_indices(single.polygons);
  const auto single_output = tf::cpp::orient_faces_consistently(single.mesh());
  CHECK(copied_indices(single_output) == single_before);
  CHECK(has_consistent_shared_edges(single_output));

  auto separate = make_mesh<Row, Ngon>(
      {Index{0}, Index{1}, Index{2}, Index{3}, Index{4}, Index{5}}, 2,
      points<Row>({0, 0, 0, 1, 0, 0, 0.5, 1, 0, 3, 0, 0, 4, 0, 0, 3.5, 1, 0}));
  const auto separate_output =
      tf::cpp::orient_faces_consistently(separate.mesh());
  CHECK(separate_output.faces_buffer().size() == 2);
  CHECK(has_consistent_shared_edges(separate_output));

  auto larger = make_mesh<Row, Ngon>(
      {Index{0}, Index{1}, Index{4}, Index{1}, Index{2}, Index{4}, Index{4},
       Index{3}, Index{2}, Index{3}, Index{0}, Index{4}},
      4, points<Row>({0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0.5, 0.5, 0}));
  REQUIRE_FALSE(has_consistent_shared_edges(larger.polygons));
  const auto larger_output = tf::cpp::orient_faces_consistently(larger.mesh());
  CHECK(larger_output.faces_buffer().size() == 4);
  CHECK(has_consistent_shared_edges(larger_output));
}

} // namespace

TEMPLATE_TEST_CASE("Python topology analysis fixtures cover every typed fixed "
                   "and dynamic mesh",
                   "[cpp][topology][python-parity][analysis][matrix][dynamic]",
                   f32_i32_2, f32_i32_3, f32_i64_2, f32_i64_3, f64_i32_2,
                   f64_i32_3, f64_i64_2, f64_i64_3) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;

  using mesh_type = tf::cpp::mesh<Index, Real, TestType::dims, 3>;

  auto (*closed_symbol)(const mesh_type &)->bool =
      &tf::cpp::is_closed<Index, Real, TestType::dims, 3>;
  auto (*manifold_symbol)(const mesh_type &)->bool =
      &tf::cpp::is_manifold<Index, Real, TestType::dims, 3>;
  auto (*edge_symbol)(const mesh_type &)->tf::cpp::nd_array<Index> =
      &tf::cpp::non_manifold_edges<Index, Real, TestType::dims, 3>;
  REQUIRE(closed_symbol != nullptr);
  REQUIRE(manifold_symbol != nullptr);
  REQUIRE(edge_symbol != nullptr);

  check_manifold_oracles_case<TestType, 3>();
  check_manifold_oracles_case<TestType, tf::dynamic_size>();
}

TEMPLATE_TEST_CASE(
    "Python manifold-edge and face-link oracles preserve fixed and jagged "
    "layout",
    "[cpp][topology][python-parity][links][matrix][dynamic][cache]", f32_i32_2,
    f32_i32_3, f32_i64_2, f32_i64_3, f64_i32_2, f64_i32_3, f64_i64_2,
    f64_i64_3) {
  using Index = typename TestType::index_type;
  using Owned = typename TestType::owned_type;

  auto fixed = two_triangles<TestType, 3>();
  const auto fixed_mesh = fixed.mesh();
  const auto fixed_faces = face_rows(fixed);
  const auto fixed_membership =
      fixed.cache.face_membership_handle(fixed_mesh.geometry());
  const auto fixed_manifold =
      tf::cpp::manifold_edge_link(fixed_faces, fixed_membership);
  CHECK(fixed_manifold.raw_shape() == tf::small_vector<int, 3>{2, 3});
  CHECK(equals(fixed_manifold, {Index{-1}, Index{1}, Index{-1}, Index{-1},
                                Index{-1}, Index{0}}));
  const auto fixed_face_link =
      tf::cpp::face_link(fixed_faces, fixed_membership);
  CHECK(equals(fixed_face_link, {Index{0}, Index{1}, Index{2}},
               {Index{1}, Index{0}}));

  auto fixed_permuted_membership = fixed_membership.deep_copy();
  std::swap(fixed_permuted_membership.data()[1],
            fixed_permuted_membership.data()[2]);
  std::swap(fixed_permuted_membership.data()[3],
            fixed_permuted_membership.data()[4]);
  CHECK(
      equals(tf::cpp::vertex_link_faces(fixed_faces, fixed_permuted_membership),
             {Index{0}, Index{2}, Index{5}, Index{8}, Index{10}},
             {Index{2}, Index{1}, Index{0}, Index{2}, Index{3}, Index{1},
              Index{0}, Index{3}, Index{1}, Index{2}}));
  CHECK(equals(
      tf::cpp::manifold_edge_link(fixed_faces, fixed_permuted_membership),
      {Index{-1}, Index{1}, Index{-1}, Index{-1}, Index{-1}, Index{0}}));
  CHECK(equals(tf::cpp::face_link(fixed_faces, fixed_permuted_membership),
               {Index{0}, Index{1}, Index{2}}, {Index{1}, Index{0}}));
  CHECK(equals(fixed_permuted_membership,
               {Index{0}, Index{1}, Index{3}, Index{5}, Index{6}},
               {Index{0}, Index{0}, Index{1}, Index{0}, Index{1}, Index{1}}));

  // A stated link is the authority and the cache keeps what it was handed.
  fixed.cache.set_manifold_edge_link(fixed_manifold.deep_copy(),
                                     fixed_mesh.geometry());
  fixed.cache.set_face_link(fixed_face_link.deep_copy(), fixed_mesh.geometry());
  CHECK(
      equals(fixed.cache.manifold_edge_link_handle(fixed_mesh.geometry()),
             {Index{-1}, Index{1}, Index{-1}, Index{-1}, Index{-1}, Index{0}}));
  CHECK(equals(fixed.cache.face_link_handle(fixed_mesh.geometry()),
               {Index{0}, Index{1}, Index{2}}, {Index{1}, Index{0}}));
  // THE DOOR reads the VALUES as well as the shape: a peer names a face this
  // reading has, or one of the three sentinels the link states in place of one
  CHECK_THROWS_AS(fixed.cache.set_manifold_edge_link(
                      make_array<Index>({Index{99}, Index{99}, Index{99},
                                         Index{99}, Index{99}, Index{99}},
                                        {2, 3}),
                      fixed_mesh.geometry()),
                  std::out_of_range);
  CHECK_THROWS_AS(fixed.cache.set_manifold_edge_link(
                      make_array<Index>({Index{-4}, Index{1}, Index{-1},
                                         Index{-1}, Index{-1}, Index{0}},
                                        {2, 3}),
                      fixed_mesh.geometry()),
                  std::out_of_range);

  auto dynamic = two_triangles<TestType, tf::dynamic_size>();
  const auto dynamic_mesh = dynamic.mesh();
  const auto dynamic_membership =
      dynamic.cache.face_membership_handle(dynamic_mesh.geometry());
  const auto dynamic_faces = face_blocks(dynamic);
  const auto dynamic_manifold =
      tf::cpp::manifold_edge_link(dynamic_faces, dynamic_membership);
  CHECK(
      equals(dynamic_manifold, {Index{0}, Index{3}, Index{6}},
             {Index{-1}, Index{1}, Index{-1}, Index{-1}, Index{-1}, Index{0}}));
  const auto dynamic_face_link =
      tf::cpp::face_link(dynamic_faces, dynamic_membership);
  CHECK(equals(dynamic_face_link, {Index{0}, Index{1}, Index{2}},
               {Index{1}, Index{0}}));

  auto dynamic_permuted_membership = dynamic_membership.deep_copy();
  std::swap(dynamic_permuted_membership.data()[1],
            dynamic_permuted_membership.data()[2]);
  std::swap(dynamic_permuted_membership.data()[3],
            dynamic_permuted_membership.data()[4]);
  CHECK(equals(
      tf::cpp::vertex_link_faces(dynamic_faces, dynamic_permuted_membership),
      {Index{0}, Index{2}, Index{5}, Index{8}, Index{10}},
      {Index{2}, Index{1}, Index{0}, Index{2}, Index{3}, Index{1}, Index{0},
       Index{3}, Index{1}, Index{2}}));
  CHECK(equals(
      tf::cpp::manifold_edge_link(dynamic_faces, dynamic_permuted_membership),
      {Index{0}, Index{3}, Index{6}},
      {Index{-1}, Index{1}, Index{-1}, Index{-1}, Index{-1}, Index{0}}));
  CHECK(equals(tf::cpp::face_link(dynamic_faces, dynamic_permuted_membership),
               {Index{0}, Index{1}, Index{2}}, {Index{1}, Index{0}}));
  CHECK(equals(dynamic_permuted_membership,
               {Index{0}, Index{1}, Index{3}, Index{5}, Index{6}},
               {Index{0}, Index{0}, Index{1}, Index{0}, Index{1}, Index{1}}));

  const auto cached_face_link =
      dynamic.cache.face_link_handle(dynamic_mesh.geometry());
  CHECK(equals(cached_face_link, {Index{0}, Index{1}, Index{2}},
               {Index{1}, Index{0}}));
  CHECK(dynamic.cache.is_face_link_fresh(dynamic_mesh.geometry()));
  // mixed faces have peers too, in the shape the free entry produces: one
  // value per face side, blocked by this reading's own faces. So what the
  // library computed is what its door takes back.
  CHECK(
      equals(dynamic.cache.manifold_edge_link_handle(dynamic_mesh.geometry()),
             {Index{0}, Index{3}, Index{6}},
             {Index{-1}, Index{1}, Index{-1}, Index{-1}, Index{-1}, Index{0}}));
  dynamic.cache.set_manifold_edge_link(
      tf::cpp::manifold_edge_link(dynamic_faces, dynamic_membership),
      dynamic_mesh.geometry());
  CHECK(
      equals(dynamic.cache.manifold_edge_link_handle(dynamic_mesh.geometry()),
             {Index{0}, Index{3}, Index{6}},
             {Index{-1}, Index{1}, Index{-1}, Index{-1}, Index{-1}, Index{0}}));
  CHECK_THROWS_AS(
      dynamic.cache.set_manifold_edge_link({}, dynamic_mesh.geometry()),
      std::invalid_argument);
  // THE DOOR reads the VALUES as well as the shape: a peer is a face of this
  // reading or one of the link's own sentinels, and a link that states fewer
  // sides than the faces have is not this reading's
  CHECK_THROWS_AS(dynamic.cache.set_manifold_edge_link(
                      make_blocks<Index>({Index{0}, Index{1}}, {Index{0}}),
                      dynamic_mesh.geometry()),
                  std::invalid_argument);
  CHECK_THROWS_AS(dynamic.cache.set_manifold_edge_link(
                      make_blocks<Index>({Index{0}, Index{2}, Index{6}},
                                         {Index{-1}, Index{1}, Index{-1},
                                          Index{-1}, Index{-1}, Index{0}}),
                      dynamic_mesh.geometry()),
                  std::invalid_argument);
  CHECK_THROWS_AS(dynamic.cache.set_manifold_edge_link(
                      make_blocks<Index>({Index{0}, Index{3}, Index{6}},
                                         {Index{99}, Index{99}, Index{99},
                                          Index{99}, Index{99}, Index{99}}),
                      dynamic_mesh.geometry()),
                  std::out_of_range);
  CHECK_THROWS_AS(dynamic.cache.set_manifold_edge_link(
                      make_blocks<Index>({Index{0}, Index{3}, Index{6}},
                                         {Index{-4}, Index{1}, Index{-1},
                                          Index{-1}, Index{-1}, Index{0}}),
                      dynamic_mesh.geometry()),
                  std::out_of_range);
  auto assigned_face_link = dynamic_face_link.deep_copy();
  dynamic.cache.set_face_link(std::move(assigned_face_link),
                              dynamic_mesh.geometry());
  // what was stated is what the cache answers with, and it built nothing of
  // its own to answer
  CHECK(equals(dynamic.cache.face_link_handle(dynamic_mesh.geometry()),
               {Index{0}, Index{1}, Index{2}}, {Index{1}, Index{0}}));
  CHECK(dynamic.cache.face_link_build_count() == 1);
  CHECK_THROWS_AS(dynamic.cache.set_face_link({}, dynamic_mesh.geometry()),
                  std::invalid_argument);
  CHECK_THROWS_AS(dynamic.cache.set_face_link(
                      make_blocks<Index>({Index{0}, Index{1}}, {Index{2}}),
                      dynamic_mesh.geometry()),
                  std::invalid_argument);
  CHECK_THROWS_AS(dynamic.cache.set_face_link(
                      make_blocks<Index>({Index{0}, Index{1}, Index{2}},
                                         {Index{1}, Index{2}}),
                      dynamic_mesh.geometry()),
                  std::out_of_range);

  auto nonzero_start =
      make_blocks<Index>({Index{0}, Index{1}, Index{2}}, {Index{1}, Index{0}});
  nonzero_start.offsets()[0] = Index{1};
  CHECK_THROWS_AS(dynamic.cache.set_face_link(std::move(nonzero_start),
                                              dynamic_mesh.geometry()),
                  std::invalid_argument);
  auto decreasing =
      make_blocks<Index>({Index{0}, Index{1}, Index{2}}, {Index{1}, Index{0}});
  decreasing.offsets()[1] = Index{2};
  decreasing.offsets()[2] = Index{1};
  CHECK_THROWS_AS(dynamic.cache.set_face_link(std::move(decreasing),
                                              dynamic_mesh.geometry()),
                  std::invalid_argument);
  auto wrong_end =
      make_blocks<Index>({Index{0}, Index{1}, Index{2}}, {Index{1}, Index{0}});
  wrong_end.offsets()[2] = Index{1};
  CHECK_THROWS_AS(dynamic.cache.set_face_link(std::move(wrong_end),
                                              dynamic_mesh.geometry()),
                  std::invalid_argument);

  tf::cpp::test::mixed_mesh_of<Owned> empty;
  auto malformed_empty = make_blocks<Index>({Index{0}}, {});
  malformed_empty.offsets()[0] = Index{1};
  CHECK_THROWS_AS(empty.cache.set_face_link(std::move(malformed_empty),
                                            empty.mesh().geometry()),
                  std::invalid_argument);
  CHECK(dynamic.cache.face_membership_build_count() == 1);

  // Exact Python four-triangle strip fixture. Its end faces have one neighbor
  // and its interior faces have two, exercising nonuniform jagged offsets.
  auto larger = make_mesh<TestType, 3>(
      {Index{0}, Index{1}, Index{2}, Index{1}, Index{3}, Index{2}, Index{2},
       Index{3}, Index{4}, Index{3}, Index{5}, Index{4}},
      4,
      points<TestType>(
          {0, 0, 0, 1, 0, 0, 0.5, 1, 0, 1.5, 1, 0, 1, 2, 0, 2, 2, 0}));
  const auto larger_mesh = larger.mesh();
  const auto larger_faces = face_rows(larger);
  const auto larger_membership =
      larger.cache.face_membership_handle(larger_mesh.geometry());
  const auto larger_manifold =
      tf::cpp::manifold_edge_link(larger_faces, larger_membership);
  CHECK(equals(larger_manifold, {Index{-1}, Index{1}, Index{-1}, Index{-1},
                                 Index{2}, Index{0}, Index{1}, Index{3},
                                 Index{-1}, Index{-1}, Index{-1}, Index{2}}));
  const auto larger_cached_manifold =
      larger.cache.manifold_edge_link_handle(larger_mesh.geometry());
  CHECK(std::equal(larger_manifold.begin(), larger_manifold.end(),
                   larger_cached_manifold.begin(),
                   larger_cached_manifold.end()));
  const auto larger_face_link =
      tf::cpp::face_link(larger_faces, larger_membership);
  CHECK(equals(larger_face_link,
               {Index{0}, Index{1}, Index{3}, Index{5}, Index{6}},
               {Index{1}, Index{0}, Index{2}, Index{1}, Index{3}, Index{2}}));
  const auto larger_cached_face_link =
      larger.cache.face_link_handle(larger_mesh.geometry());
  CHECK(equals(larger_cached_face_link,
               {Index{0}, Index{1}, Index{3}, Index{5}, Index{6}},
               {Index{1}, Index{0}, Index{2}, Index{1}, Index{3}, Index{2}}));
}

TEMPLATE_TEST_CASE(
    "raw topology membership and vertex links match mesh authorities exactly",
    "[cpp][topology][python-parity][raw][membership][vertex-link][cache]",
    f32_i32_3, f64_i64_3) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using Owned = typename TestType::owned_type;

  auto edges = make_array<Index>(
      {Index{0}, Index{1}, Index{1}, Index{2}, Index{2}, Index{3}}, {3, 2});
  tf::cpp::test::owned_edge_mesh<Index, Real, 3> edge_owner{
      tf::cpp::test::segments_of<Index, Real, 3>(
          std::vector<Index>{Index{0}, Index{1}, Index{1}, Index{2}, Index{2},
                             Index{3}},
          points<TestType>({0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0}))};
  const auto edge_mesh = edge_owner.edge_mesh();
  const auto edge_membership = tf::cpp::cell_membership(edges, 4);
  const auto edge_vertex_link = tf::cpp::vertex_link_edges(edges, 4);
  STATIC_REQUIRE(
      std::is_same_v<decltype(edge_membership),
                     const tf::cpp::offset_blocked_buffer<Index, Index>>);
  CHECK(equals(edge_membership,
               {Index{0}, Index{1}, Index{3}, Index{5}, Index{6}},
               {Index{0}, Index{1}, Index{0}, Index{2}, Index{1}, Index{2}}));
  CHECK(equals(edge_vertex_link,
               {Index{0}, Index{1}, Index{3}, Index{5}, Index{6}},
               {Index{1}, Index{2}, Index{0}, Index{3}, Index{1}, Index{2}}));
  CHECK(equals(edge_membership,
               edge_owner.cache.edge_membership_handle(edge_mesh.geometry())));
  CHECK(equals(edge_vertex_link,
               edge_owner.cache.vertex_link_handle(edge_mesh.geometry())));

  auto fixed = two_triangles<TestType, 3>();
  const auto fixed_mesh = fixed.mesh();
  const auto fixed_faces = face_rows(fixed);
  const auto fixed_membership = tf::cpp::cell_membership(fixed_faces, Index{4});
  const auto fixed_vertex_link =
      tf::cpp::vertex_link_faces(fixed_faces, fixed_membership);
  CHECK(equals(fixed_membership,
               {Index{0}, Index{1}, Index{3}, Index{5}, Index{6}},
               {Index{0}, Index{1}, Index{0}, Index{1}, Index{0}, Index{1}}));
  CHECK(equals(fixed_vertex_link,
               {Index{0}, Index{2}, Index{5}, Index{8}, Index{10}},
               {Index{2}, Index{1}, Index{2}, Index{3}, Index{0}, Index{3},
                Index{1}, Index{0}, Index{1}, Index{2}}));
  CHECK(equals(fixed_membership,
               fixed.cache.face_membership_handle(fixed_mesh.geometry())));
  CHECK(equals(fixed_vertex_link,
               fixed.cache.vertex_link_handle(fixed_mesh.geometry())));

  tf::cpp::test::mixed_mesh_of<Owned> dynamic{
      tf::cpp::test::polygons_of<Index, Real, TestType::dims>(
          std::vector<Index>{Index{0}, Index{3}, Index{7}},
          std::vector<Index>{Index{0}, Index{1}, Index{2}, Index{1}, Index{3},
                             Index{4}, Index{2}},
          points<TestType>({0, 0, 0, 1, 0, 0, 0.5, 1, 0, 2, 0, 0, 1.5, 1, 0}))};
  const auto dynamic_mesh = dynamic.mesh();
  const auto dynamic_faces = face_blocks(dynamic);
  const auto dynamic_membership =
      tf::cpp::cell_membership(dynamic_faces, Index{5});
  const auto dynamic_vertex_link =
      tf::cpp::vertex_link_faces(dynamic_faces, dynamic_membership);
  CHECK(equals(
      dynamic_membership,
      {Index{0}, Index{1}, Index{3}, Index{5}, Index{6}, Index{7}},
      {Index{0}, Index{1}, Index{0}, Index{1}, Index{0}, Index{1}, Index{1}}));
  CHECK(equals(dynamic_vertex_link,
               {Index{0}, Index{2}, Index{5}, Index{8}, Index{10}, Index{12}},
               {Index{2}, Index{1}, Index{2}, Index{3}, Index{0}, Index{4},
                Index{1}, Index{0}, Index{1}, Index{4}, Index{3}, Index{2}}));
  CHECK(equals(dynamic_membership,
               dynamic.cache.face_membership_handle(dynamic_mesh.geometry())));
  CHECK(equals(dynamic_vertex_link,
               dynamic.cache.vertex_link_handle(dynamic_mesh.geometry())));
}

TEMPLATE_TEST_CASE(
    "raw topology empty results are canonical and async inputs are owned",
    "[cpp][topology][python-parity][raw][async][ownership][empty]", f32_i32_3,
    f64_i64_3) {
  using Index = typename TestType::index_type;

  auto empty_edges = make_array<Index>({}, {0, 2});
  auto empty_faces = make_array<Index>({}, {0, 3});
  auto empty_dynamic = make_blocks<Index>({Index{0}}, {});
  auto offsetless_dynamic =
      tf::cpp::offset_blocked_buffer<Index, Index>::create(
          make_array<Index>({}, {0}), make_array<Index>({}, {0}));
  const auto empty_edge_membership =
      tf::cpp::cell_membership(empty_edges, Index{0});
  const auto empty_face_membership =
      tf::cpp::cell_membership(empty_faces, Index{0});
  CHECK(equals(empty_edge_membership, {Index{0}}, {}));
  CHECK(equals(empty_face_membership, {Index{0}}, {}));
  CHECK(equals(tf::cpp::cell_membership(empty_dynamic, Index{0}), {Index{0}},
               {}));
  CHECK_THROWS_AS(tf::cpp::cell_membership(offsetless_dynamic, Index{0}),
                  std::invalid_argument);
  // a structure over a declared identity space has one fence per id and one
  // more; a structure generated from a range has as many as the range gave it
  CHECK(equals(tf::cpp::vertex_link_edges(empty_edges, Index{0}), {Index{0}},
               {}));
  CHECK(equals(tf::cpp::vertex_link_faces(empty_faces, empty_face_membership),
               {}, {}));
  CHECK(equals(tf::cpp::vertex_link_faces(empty_dynamic, empty_face_membership),
               {}, {}));
  CHECK_THROWS_AS(tf::cpp::vertex_link_faces(empty_faces, offsetless_dynamic),
                  std::invalid_argument);

  auto edges =
      make_array<Index>({Index{0}, Index{1}, Index{1}, Index{2}}, {2, 2});
  auto faces = make_array<Index>(
      {Index{0}, Index{1}, Index{2}, Index{1}, Index{3}, Index{2}}, {2, 3});
  auto dynamic_faces = make_blocks<Index>(
      {Index{0}, Index{3}, Index{7}},
      {Index{0}, Index{1}, Index{2}, Index{1}, Index{3}, Index{4}, Index{2}});
  auto fixed_membership = tf::cpp::cell_membership(faces, Index{4});
  auto dynamic_membership = tf::cpp::cell_membership(dynamic_faces, Index{5});

  auto pending_edge_membership =
      tf::cpp::async::cell_membership(edges, Index{3});
  auto pending_dynamic_membership =
      tf::cpp::async::cell_membership(dynamic_faces, Index{5});
  auto pending_edge_link = tf::cpp::async::vertex_link_edges(edges, Index{3});
  auto pending_fixed_link =
      tf::cpp::async::vertex_link_faces(faces, fixed_membership);
  auto pending_dynamic_link =
      tf::cpp::async::vertex_link_faces(dynamic_faces, dynamic_membership);
  edges.destroy();
  faces.destroy();
  dynamic_faces.destroy();
  fixed_membership.destroy();
  dynamic_membership.destroy();

  CHECK(equals(pending_edge_membership.get(),
               {Index{0}, Index{1}, Index{3}, Index{4}},
               {Index{0}, Index{1}, Index{0}, Index{1}}));
  CHECK(equals(
      pending_dynamic_membership.get(),
      {Index{0}, Index{1}, Index{3}, Index{5}, Index{6}, Index{7}},
      {Index{0}, Index{1}, Index{0}, Index{1}, Index{0}, Index{1}, Index{1}}));
  CHECK(equals(pending_edge_link.get(),
               {Index{0}, Index{1}, Index{3}, Index{4}},
               {Index{1}, Index{2}, Index{0}, Index{1}}));
  CHECK(equals(pending_fixed_link.get(),
               {Index{0}, Index{2}, Index{5}, Index{8}, Index{10}},
               {Index{2}, Index{1}, Index{2}, Index{3}, Index{0}, Index{3},
                Index{1}, Index{0}, Index{1}, Index{2}}));
  CHECK(equals(pending_dynamic_link.get(),
               {Index{0}, Index{2}, Index{5}, Index{8}, Index{10}, Index{12}},
               {Index{2}, Index{1}, Index{2}, Index{3}, Index{0}, Index{4},
                Index{1}, Index{0}, Index{1}, Index{4}, Index{3}, Index{2}}));
}

TEMPLATE_TEST_CASE("raw topology carriers reject malformed rank arity offsets "
                   "domains and blocks",
                   "[cpp][topology][python-parity][raw][validation]",
                   std::int32_t, std::int64_t) {
  using Index = TestType;

  CHECK_THROWS_AS(
      tf::cpp::cell_membership(
          make_array<Index>({Index{0}, Index{1}, Index{2}}, {3}), Index{3}),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::cell_membership(
          make_array<Index>({Index{0}, Index{1}, Index{2}, Index{3}}, {1, 4}),
          Index{4}),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::cell_membership(make_array<Index>({Index{0}, Index{1}}, {1, 2}),
                               Index{-1}),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::cell_membership(make_array<Index>({Index{0}, Index{3}}, {1, 2}),
                               Index{3}),
      std::out_of_range);
  CHECK_THROWS_AS(
      tf::cpp::vertex_link_edges(
          make_array<Index>({Index{0}, Index{1}, Index{2}}, {1, 3}), Index{3}),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::vertex_link_edges(
          make_array<Index>({Index{0}, Index{-1}}, {1, 2}), Index{2}),
      std::out_of_range);

  auto nonzero =
      make_blocks<Index>({Index{0}, Index{3}}, {Index{0}, Index{1}, Index{2}});
  nonzero.offsets()[0] = Index{1};
  CHECK_THROWS_AS(tf::cpp::cell_membership(nonzero, Index{3}),
                  std::invalid_argument);
  auto decreasing = make_blocks<Index>(
      {Index{0}, Index{3}, Index{6}},
      {Index{0}, Index{1}, Index{2}, Index{0}, Index{2}, Index{3}});
  decreasing.offsets()[1] = Index{5};
  decreasing.offsets()[2] = Index{4};
  CHECK_THROWS_AS(tf::cpp::cell_membership(decreasing, Index{4}),
                  std::invalid_argument);
  auto wrong_end =
      make_blocks<Index>({Index{0}, Index{3}}, {Index{0}, Index{1}, Index{2}});
  wrong_end.offsets()[1] = Index{2};
  CHECK_THROWS_AS(tf::cpp::cell_membership(wrong_end, Index{3}),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::cell_membership(
          make_blocks<Index>({Index{0}, Index{2}}, {Index{0}, Index{1}}),
          Index{2}),
      std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::cell_membership(
                      make_blocks<Index>({Index{0}, Index{3}},
                                         {Index{0}, Index{1}, Index{3}}),
                      Index{3}),
                  std::out_of_range);

  auto faces = make_array<Index>(
      {Index{0}, Index{1}, Index{2}, Index{1}, Index{3}, Index{2}}, {2, 3});
  auto membership = tf::cpp::cell_membership(faces, Index{4});
  CHECK_THROWS_AS(
      tf::cpp::vertex_link_faces(
          make_array<Index>({Index{0}, Index{1}, Index{2}}, {3}), membership),
      std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::vertex_link_faces(
                      make_array<Index>({Index{0}, Index{1}}, {1, 2}),
                      make_blocks<Index>({Index{0}, Index{1}, Index{2}},
                                         {Index{0}, Index{0}})),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::vertex_link_faces(
                      make_array<Index>({Index{0}, Index{1}, Index{4}}, {1, 3}),
                      membership),
                  std::out_of_range);
  auto bad_offsets = membership.deep_copy();
  bad_offsets.offsets()[1] = Index{-1};
  CHECK_THROWS_AS(tf::cpp::vertex_link_faces(faces, bad_offsets),
                  std::invalid_argument);
  auto bad_face_id = membership.deep_copy();
  bad_face_id.data()[0] = Index{2};
  CHECK_THROWS_AS(tf::cpp::vertex_link_faces(faces, bad_face_id),
                  std::out_of_range);
  auto inconsistent = membership.deep_copy();
  inconsistent.data()[1] = Index{0};
  CHECK_THROWS_AS(tf::cpp::vertex_link_faces(faces, inconsistent),
                  std::invalid_argument);

  auto repeated_faces = make_array<Index>(
      {Index{0}, Index{1}, Index{0}, Index{0}, Index{2}, Index{3}}, {2, 3});
  auto wrong_multiplicity = tf::cpp::cell_membership(repeated_faces, Index{4});
  REQUIRE(equals(wrong_multiplicity,
                 {Index{0}, Index{3}, Index{4}, Index{5}, Index{6}},
                 {Index{1}, Index{0}, Index{0}, Index{0}, Index{1}, Index{1}}));
  wrong_multiplicity.data()[1] = Index{1};
  CHECK_THROWS_AS(
      tf::cpp::vertex_link_faces(repeated_faces, wrong_multiplicity),
      std::invalid_argument);

  CHECK_THROWS_AS(
      tf::cpp::vertex_link_faces(
          make_blocks<Index>({Index{0}, Index{2}}, {Index{0}, Index{1}}),
          make_blocks<Index>({Index{0}, Index{1}, Index{2}},
                             {Index{0}, Index{0}})),
      std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "Python consistent-orientation fixtures preserve every carrier axis",
    "[cpp][topology][python-parity][orientation][matrix][dynamic][ownership]",
    f32_i32_2, f32_i32_3, f32_i64_2, f32_i64_3, f64_i32_2, f64_i32_3, f64_i64_2,
    f64_i64_3) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using function_type = tf::polygons_buffer<Index, Real, TestType::dims, 3> (*)(
      const tf::cpp::mesh<Index, Real, TestType::dims, 3> &);
  const function_type symbol =
      &tf::cpp::orient_faces_consistently<Index, Real, TestType::dims, 3>;
  REQUIRE(symbol != nullptr);

  check_orient_faces_case<TestType, 3>();
  check_orient_faces_case<TestType, tf::dynamic_size>();
}

TEST_CASE("generalized topology async calls submit once and read the mesh they "
          "were handed",
          "[cpp][topology][python-parity][analysis][orientation][async]") {
  using Row = f64_i64_2;
  using Index = typename Row::index_type;
  auto submissions = std::make_shared<std::atomic<int>>(0);

  // A mesh is one coherent reading of memory the caller keeps, so it is
  // carried to the executor as it stands; a resolver that assembles ANOTHER
  // mesh states nothing about the pending one.
  auto input = two_triangles<Row, tf::dynamic_size>(true);
  auto other = tetrahedron<Row, tf::dynamic_size>();
  auto pending_orientation = tf::cpp::async::orient_faces_consistently(
      mutating_resolver{submissions, [&] { static_cast<void>(other.mesh()); }},
      input.mesh());
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  auto oriented = pending_orientation.get();
  CHECK(oriented.faces_buffer().size() == 2);
  CHECK(has_consistent_shared_edges(oriented));

  auto open = two_triangles<Row, tf::dynamic_size>();
  auto pending_closed = tf::cpp::async::is_closed(
      mutating_resolver{submissions, [&] { static_cast<void>(other.mesh()); }},
      open.mesh());
  CHECK(submissions->load(std::memory_order_relaxed) == 2);
  CHECK_FALSE(pending_closed.get());

  // A raw array is a handle, so the entry that takes one retains it and the
  // caller may drop its own.
  auto links_mesh = two_triangles<Row, tf::dynamic_size>();
  auto faces = face_blocks(links_mesh);
  auto membership =
      links_mesh.cache.face_membership_handle(links_mesh.mesh().geometry());
  auto pending_links = tf::cpp::async::manifold_edge_link(faces, membership);
  faces.destroy();
  membership.destroy();
  links_mesh = {};
  const auto links = pending_links.get();
  CHECK(
      equals(links, {Index{0}, Index{3}, Index{6}},
             {Index{-1}, Index{1}, Index{-1}, Index{-1}, Index{-1}, Index{0}}));
}

TEST_CASE("topology computational getters validate malformed typed carriers",
          "[cpp][topology][python-parity][validation]") {
  using Row = f32_i64_3;
  using Index = typename Row::index_type;

  // a caller holding nothing holds the EMPTY mesh, which is closed and already
  // oriented
  typename Row::owned_type nothing;
  CHECK(tf::cpp::is_closed(nothing.mesh()));
  CHECK(tf::cpp::orient_faces_consistently(nothing.mesh())
            .faces_buffer()
            .size() == 0);

  // the library cannot see a caller's storage, so a face index is refused
  // where the READING is read, once, and remembered for it
  auto bad = two_triangles<Row, 3>();
  tf::cpp::test::fill_storage(bad.polygons.points_buffer().data_buffer(),
                              points<Row>({0, 0, 0, 1, 0, 0}));
  bad.cache.points_changed();
  CHECK_THROWS_AS(tf::cpp::is_manifold(bad.mesh()), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::non_manifold_edges(bad.mesh()), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::orient_faces_consistently(bad.mesh()),
                  std::out_of_range);

  auto source = two_triangles<Row, 3>();
  auto membership =
      source.cache.face_membership_handle(source.mesh().geometry());
  CHECK_THROWS_AS(tf::cpp::manifold_edge_link(
                      make_array<Index>({0, 1, 2, 3}, {1, 4}), membership),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::face_link(make_array<Index>({0, 1, 2, 3}, {1, 4}), membership),
      std::invalid_argument);

  auto empty_faces = tf::cpp::offset_blocked_buffer<Index, Index>::create(
      make_array<Index>({Index{0}}, {1}), make_array<Index>({}, {0}));
  auto empty_membership = tf::cpp::offset_blocked_buffer<Index, Index>::create(
      make_array<Index>({Index{0}}, {1}), make_array<Index>({}, {0}));
  const auto empty_manifold =
      tf::cpp::manifold_edge_link(empty_faces, empty_membership);
  CHECK(equals(empty_manifold, {}, {}));
  // every link is the core's structure verbatim, and no faces is the empty one
  for (int repetition = 0; repetition < 32; ++repetition) {
    auto fresh_faces = empty_faces.deep_copy();
    auto fresh_membership = empty_membership.deep_copy();
    const auto empty_face_link =
        tf::cpp::face_link(fresh_faces, fresh_membership);
    CHECK(equals(empty_face_link, {}, {}));
  }
}

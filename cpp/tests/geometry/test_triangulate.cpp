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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/geometry/async/triangulate.hpp"
#include "trueform/cpp/geometry/triangulate.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename Index, typename Real, std::size_t Dims> struct matrix_row {
  using real_type = Real;
  using index_type = Index;
  static constexpr std::size_t dims = Dims;
};

using matrix_rows = std::tuple<
    matrix_row<std::int32_t, float, 2>, matrix_row<std::int32_t, float, 3>,
    matrix_row<std::int64_t, float, 2>, matrix_row<std::int64_t, float, 3>,
    matrix_row<std::int32_t, double, 2>, matrix_row<std::int32_t, double, 3>,
    matrix_row<std::int64_t, double, 2>, matrix_row<std::int64_t, double, 3>>;

struct counting_resolver {
  int *count;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    ++*count;
    return std::make_shared<state_type<T>>();
  }
};

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

template <typename T>
auto make_array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(values.size());
  std::copy(values.begin(), values.end(), buffer.begin());
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename T>
auto make_array(const std::vector<T> &values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(values.size());
  std::copy(values.begin(), values.end(), buffer.begin());
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename Real, std::size_t Dims>
auto points(std::initializer_list<std::array<Real, Dims>> values)
    -> tf::cpp::nd_array<Real> {
  std::vector<Real> flat;
  flat.reserve(values.size() * Dims);
  for (const auto &point : values)
    flat.insert(flat.end(), point.begin(), point.end());
  return make_array(flat,
                    {static_cast<int>(values.size()), static_cast<int>(Dims)});
}

template <typename Real, std::size_t Dims>
auto regular_polygon(int count, bool clockwise = false)
    -> tf::cpp::nd_array<Real> {
  const auto pi = std::acos(-1.0);
  std::vector<Real> flat;
  flat.reserve(static_cast<std::size_t>(count) * Dims);
  for (int index = 0; index < count; ++index) {
    const auto direction = clockwise ? -1.0 : 1.0;
    const auto angle = direction * 2.0 * pi * index / count;
    flat.push_back(static_cast<Real>(std::cos(angle)));
    flat.push_back(static_cast<Real>(std::sin(angle)));
    if constexpr (Dims == 3)
      flat.push_back(Real{0});
  }
  return make_array(flat, {count, static_cast<int>(Dims)});
}

template <typename Index>
auto dynamic_faces(std::initializer_list<Index> offsets,
                   std::initializer_list<Index> data)
    -> tf::cpp::offset_blocked_buffer<Index, Index> {
  return tf::cpp::offset_blocked_buffer<Index, Index>::create(
      make_array<Index>(offsets, {static_cast<int>(offsets.size())}),
      make_array<Index>(data, {static_cast<int>(data.size())}));
}

template <typename Real, std::size_t Dims>
auto shared_quad_points() -> tf::cpp::nd_array<Real> {
  if constexpr (Dims == 2)
    return points<Real, 2>(
        {{{0, 0}}, {{1, 0}}, {{1, 1}}, {{0, 1}}, {{2, 0}}, {{2, 1}}});
  else
    return points<Real, 3>({{{0, 0, 0}},
                            {{1, 0, 0}},
                            {{1, 1, 0}},
                            {{0, 1, 0}},
                            {{2, 0, 0}},
                            {{2, 1, 0}}});
}

template <typename Real, std::size_t Dims>
auto square_points() -> tf::cpp::nd_array<Real> {
  if constexpr (Dims == 2)
    return points<Real, 2>({{{0, 0}}, {{1, 0}}, {{1, 1}}, {{0, 1}}});
  else
    return points<Real, 3>(
        {{{0, 0, 0}}, {{1, 0, 0}}, {{1, 1, 0}}, {{0, 1, 0}}});
}

template <typename Real, std::size_t Dims>
auto x_stretch() -> std::array<Real, (Dims + 1) * (Dims + 1)> {
  if constexpr (Dims == 2)
    return {10, 0, 0, 0, 1, 0, 0, 0, 1};
  else
    return {10, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}

template <typename Index, typename Real, std::size_t Dims>
using triangulated_t = tf::polygons_buffer<Index, Real, Dims, 3>;

template <typename Index, typename Real, std::size_t Dims>
using mixed_mesh_t =
    tf::cpp::test::owned_mesh<Index, Real, Dims, tf::dynamic_size>;

template <typename Index, typename Real, std::size_t Dims>
using fixed_mesh_t = tf::cpp::test::owned_mesh<Index, Real, Dims>;

/// A triangulation is core's own storage, so a check reads its flat arrays
/// where they lie.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto faces_of(const tf::polygons_buffer<Index, Real, Dims, Ngon> &value)
    -> const tf::buffer<Index> & {
  return value.faces_buffer().data_buffer();
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto points_of(const tf::polygons_buffer<Index, Real, Dims, Ngon> &value)
    -> const tf::buffer<Real> & {
  return value.points_buffer().data_buffer();
}

template <typename T>
auto copied(const tf::buffer<T> &values) -> std::vector<T> {
  return {values.data(), values.data() + values.size()};
}

template <typename Index, typename Real, std::size_t Dims>
auto dynamic_mesh() -> mixed_mesh_t<Index, Real, Dims> {
  const auto points = shared_quad_points<Real, Dims>();
  mixed_mesh_t<Index, Real, Dims> owned;
  owned.polygons =
      tf::cpp::test::polygons_of<Index, Real, Dims, std::vector<Index>,
                                 std::vector<Index>, std::vector<Real>>(
          {0, 4, 8}, {0, 1, 2, 3, 1, 4, 5, 2}, {points.begin(), points.end()});
  return owned;
}

template <typename T>
auto check_equal(const tf::cpp::nd_array<T> &actual,
                 const tf::cpp::nd_array<T> &expected) -> void {
  REQUIRE(actual.raw_shape() == expected.raw_shape());
  REQUIRE(actual.length() == expected.length());
  for (std::size_t index = 0; index < actual.length(); ++index)
    CHECK(actual[index] == expected[index]);
}

template <typename T>
auto check_equal(const tf::buffer<T> &actual,
                 const tf::cpp::nd_array<T> &expected) -> void {
  REQUIRE(actual.size() == expected.length());
  for (std::size_t index = 0; index < actual.size(); ++index)
    CHECK(actual[index] == expected[index]);
}

template <typename T>
auto check_equal(const tf::buffer<T> &actual, const std::vector<T> &expected)
    -> void {
  REQUIRE(actual.size() == expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index)
    CHECK(actual[index] == expected[index]);
}

template <typename T>
auto check_equal(const tf::buffer<T> &actual, const tf::buffer<T> &expected)
    -> void {
  REQUIRE(actual.size() == expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index)
    CHECK(actual[index] == expected[index]);
}

template <typename Index, typename Real, std::size_t Dims>
auto check_triangle_mesh(const triangulated_t<Index, Real, Dims> &value,
                         int faces, int point_count) -> void {
  CHECK(static_cast<int>(value.faces_buffer().size()) == faces);
  CHECK(static_cast<int>(value.points_buffer().size()) == point_count);
  CHECK(faces_of(value).size() == static_cast<std::size_t>(faces) * 3);
  CHECK(points_of(value).size() ==
        static_cast<std::size_t>(point_count) * Dims);
}

template <typename Index, typename Real, std::size_t Dims>
auto signed_xy_area(const triangulated_t<Index, Real, Dims> &value) -> double {
  double area = 0;
  const auto &faces = faces_of(value);
  const auto &vertices = points_of(value);
  for (int face = 0; face < static_cast<int>(value.faces_buffer().size());
       ++face) {
    const auto i0 = static_cast<std::size_t>(faces[3 * face]);
    const auto i1 = static_cast<std::size_t>(faces[3 * face + 1]);
    const auto i2 = static_cast<std::size_t>(faces[3 * face + 2]);
    const auto x0 = static_cast<double>(vertices[i0 * Dims]);
    const auto y0 = static_cast<double>(vertices[i0 * Dims + 1]);
    const auto x1 = static_cast<double>(vertices[i1 * Dims]);
    const auto y1 = static_cast<double>(vertices[i1 * Dims + 1]);
    const auto x2 = static_cast<double>(vertices[i2 * Dims]);
    const auto y2 = static_cast<double>(vertices[i2 * Dims + 1]);
    area += 0.5 * ((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0));
  }
  return area;
}

template <typename Index, typename Real, std::size_t Dims>
auto require_valid_nondegenerate(const triangulated_t<Index, Real, Dims> &value)
    -> void {
  const auto &faces = faces_of(value);
  const auto &vertices = points_of(value);
  for (std::size_t index = 0; index < faces.size(); ++index) {
    CHECK(faces[index] >= Index{0});
    CHECK(faces[index] <
          static_cast<Index>(static_cast<int>(value.points_buffer().size())));
  }
  for (int face = 0; face < static_cast<int>(value.faces_buffer().size());
       ++face) {
    const auto i0 = static_cast<std::size_t>(faces[3 * face]);
    const auto i1 = static_cast<std::size_t>(faces[3 * face + 1]);
    const auto i2 = static_cast<std::size_t>(faces[3 * face + 2]);
    const auto x0 = static_cast<double>(vertices[i0 * Dims]);
    const auto y0 = static_cast<double>(vertices[i0 * Dims + 1]);
    const auto x1 = static_cast<double>(vertices[i1 * Dims]);
    const auto y1 = static_cast<double>(vertices[i1 * Dims + 1]);
    const auto x2 = static_cast<double>(vertices[i2 * Dims]);
    const auto y2 = static_cast<double>(vertices[i2 * Dims + 1]);
    CHECK(std::abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)) > 1e-12);
  }
}

} // namespace

TEMPLATE_LIST_TEST_CASE(
    "typed polygon-array triangulation covers every Python literal fixture",
    "[cpp][geometry][triangulate][python-parity][polygon][matrix]",
    matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;
  using mesh_type = triangulated_t<Index, Real, Dims>;

  auto quad = square_points<Real, Dims>();
  const auto quad_before = quad.deep_copy();
  const auto *quad_storage = quad.raw_data();
  auto quad_result = tf::cpp::triangulate<Index, Real, Dims>(quad);
  STATIC_REQUIRE(std::is_same_v<decltype(quad_result), mesh_type>);
  check_triangle_mesh(quad_result, 2, 4);
  check_equal(points_of(quad_result), quad_before);
  CHECK(static_cast<const void *>(points_of(quad_result).data()) !=
        static_cast<const void *>(quad_storage));
  check_equal(quad, quad_before);
  CHECK(signed_xy_area(quad_result) == Catch::Approx(1.0));

  auto pentagon = regular_polygon<Real, Dims>(5);
  auto pentagon_result = tf::cpp::triangulate<Index, Real, Dims>(pentagon);
  check_triangle_mesh(pentagon_result, 3, 5);

  auto python_2d_pentagon = [&] {
    if constexpr (Dims == 2)
      return points<Real, 2>(
          {{{0, 0}}, {{1, 0}}, {{1, 1}}, {{Real{0.5}, Real{1.5}}}, {{0, 1}}});
    else
      return points<Real, 3>({{{0, 0, 0}},
                              {{1, 0, 0}},
                              {{1, 1, 0}},
                              {{Real{0.5}, Real{1.5}, 0}},
                              {{0, 1, 0}}});
  }();
  auto python_2d_result =
      tf::cpp::triangulate<Index, Real, Dims>(python_2d_pentagon);
  check_triangle_mesh(python_2d_result, 3, 5);
  require_valid_nondegenerate(python_2d_result);

  auto triangle = [&] {
    if constexpr (Dims == 2)
      return points<Real, 2>({{{0, 0}}, {{1, 0}}, {{Real{0.5}, 1}}});
    else
      return points<Real, 3>({{{0, 0, 0}}, {{1, 0, 0}}, {{Real{0.5}, 1, 0}}});
  }();
  auto triangle_result = tf::cpp::triangulate<Index, Real, Dims>(triangle);
  check_triangle_mesh(triangle_result, 1, 3);
  check_equal(points_of(triangle_result), triangle);

  // Literal from test_single_polygon_dtype_preserved.
  auto dtype_polygon = [&] {
    if constexpr (Dims == 2)
      return points<Real, 2>(
          {{{0, 0}}, {{1, 0}}, {{Real{0.5}, 1}}, {{0, Real{0.5}}}});
    else
      return points<Real, 3>(
          {{{0, 0, 0}}, {{1, 0, 0}}, {{Real{0.5}, 1, 0}}, {{0, Real{0.5}, 0}}});
  }();
  auto dtype_result = tf::cpp::triangulate<Index, Real, Dims>(dtype_polygon);
  check_triangle_mesh(dtype_result, 2, 4);
  check_equal(points_of(dtype_result), dtype_polygon);

  std::vector<Real> batch_values;
  const std::array<std::array<Real, 2>, 8> batch_xy{{
      {{0, 0}},
      {{1, 0}},
      {{1, 1}},
      {{0, 1}},
      {{2, 0}},
      {{3, 0}},
      {{3, 1}},
      {{2, 1}},
  }};
  for (const auto &point : batch_xy) {
    batch_values.push_back(point[0]);
    batch_values.push_back(point[1]);
    if constexpr (Dims == 3)
      batch_values.push_back(Real{0});
  }
  auto batch = make_array(batch_values, {2, 4, static_cast<int>(Dims)});
  auto batch_result = tf::cpp::triangulate<Index, Real, Dims>(batch);
  check_triangle_mesh(batch_result, 4, 8);
  CHECK(static_cast<const void *>(points_of(batch_result).data()) !=
        static_cast<const void *>(batch.raw_data()));
}

TEMPLATE_LIST_TEST_CASE(
    "typed fixed faces and mesh inputs match triangle quad and grid fixtures",
    "[cpp][geometry][triangulate][python-parity][fixed][mesh][matrix]",
    matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  auto points_in = square_points<Real, Dims>();
  auto triangle_faces = make_array<Index>({0, 1, 2, 0, 2, 3}, {2, 3});
  const auto *face_storage = triangle_faces.raw_data();
  const auto *point_storage = points_in.raw_data();
  auto triangle_result =
      tf::cpp::triangulate<Index, Real, Dims>(triangle_faces, points_in);
  check_triangle_mesh(triangle_result, 2, 4);
  check_equal(faces_of(triangle_result), triangle_faces);
  check_equal(points_of(triangle_result), points_in);
  CHECK(static_cast<const void *>(faces_of(triangle_result).data()) !=
        static_cast<const void *>(face_storage));
  CHECK(static_cast<const void *>(points_of(triangle_result).data()) !=
        static_cast<const void *>(point_storage));

  fixed_mesh_t<Index, Real, Dims> triangle_mesh;
  triangle_mesh.polygons =
      tf::cpp::test::polygons_of<Index, Real, Dims, std::vector<Index>,
                                 std::vector<Real>>(
          {0, 1, 2, 0, 2, 3}, {points_in.begin(), points_in.end()});
  const auto mesh_faces = copied(faces_of(triangle_mesh.polygons));
  const auto mesh_points = copied(points_of(triangle_mesh.polygons));
  auto mesh_result =
      tf::cpp::triangulate<Index, Real, Dims>(triangle_mesh.mesh());
  auto inferred_mesh_result = tf::cpp::triangulate(triangle_mesh.mesh());
  check_triangle_mesh(mesh_result, 2, 4);
  check_triangle_mesh(inferred_mesh_result, 2, 4);
  check_equal(faces_of(mesh_result), mesh_faces);
  check_equal(points_of(mesh_result), mesh_points);
  CHECK(faces_of(mesh_result).data() !=
        faces_of(triangle_mesh.polygons).data());
  CHECK(points_of(mesh_result).data() !=
        points_of(triangle_mesh.polygons).data());

  auto quad_faces = make_array<Index>({0, 1, 2, 3}, {1, 4});
  const auto quad_before = quad_faces.deep_copy();
  auto quad_result =
      tf::cpp::triangulate<Index, Real, Dims>(quad_faces, points_in);
  check_triangle_mesh(quad_result, 2, 4);
  check_equal(quad_faces, quad_before);
  CHECK(signed_xy_area(quad_result) == Catch::Approx(1.0));

  auto two_quads = make_array<Index>({0, 1, 2, 3, 1, 4, 5, 2}, {2, 4});
  auto shared_points = shared_quad_points<Real, Dims>();
  auto two_quad_result =
      tf::cpp::triangulate<Index, Real, Dims>(two_quads, shared_points);
  check_triangle_mesh(two_quad_result, 4, 6);
  CHECK(signed_xy_area(two_quad_result) == Catch::Approx(2.0));

  std::vector<Real> grid_points;
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) {
      grid_points.push_back(static_cast<Real>(x));
      grid_points.push_back(static_cast<Real>(y));
      if constexpr (Dims == 3)
        grid_points.push_back(Real{0});
    }
  std::vector<Index> grid_faces;
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x) {
      const auto v0 = static_cast<Index>(y * 4 + x);
      grid_faces.insert(grid_faces.end(), {v0, static_cast<Index>(v0 + 1),
                                           static_cast<Index>(v0 + 5),
                                           static_cast<Index>(v0 + 4)});
    }
  auto grid = make_array(grid_points, {16, static_cast<int>(Dims)});
  auto quads = make_array(grid_faces, {9, 4});
  auto grid_result = tf::cpp::triangulate<Index, Real, Dims>(quads, grid);
  check_triangle_mesh(grid_result, 18, 16);
  CHECK(signed_xy_area(grid_result) == Catch::Approx(9.0));
}

TEMPLATE_LIST_TEST_CASE(
    "typed dynamic face arrays and dynamic mesh inputs match Python fixtures",
    "[cpp][geometry][triangulate][python-parity][dynamic][matrix]",
    matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  auto faces = dynamic_faces<Index>({0, 4, 8}, {0, 1, 2, 3, 1, 4, 5, 2});
  auto point_array = shared_quad_points<Real, Dims>();
  const auto offsets_before = faces.offsets().deep_copy();
  const auto data_before = faces.data().deep_copy();
  const auto points_before = point_array.deep_copy();
  auto result = tf::cpp::triangulate<Index, Real, Dims>(faces, point_array);
  check_triangle_mesh(result, 4, 6);
  check_equal(faces.offsets(), offsets_before);
  check_equal(faces.data(), data_before);
  check_equal(point_array, points_before);
  CHECK(static_cast<const void *>(points_of(result).data()) !=
        static_cast<const void *>(point_array.raw_data()));
  CHECK(signed_xy_area(result) == Catch::Approx(2.0));

  std::vector<Real> radial_points;
  const auto pi = std::acos(-1.0);
  for (int index = 0; index < 12; ++index) {
    const auto angle = 2.0 * pi * index / 12.0;
    radial_points.push_back(static_cast<Real>(std::cos(angle) * (index + 1)));
    radial_points.push_back(static_cast<Real>(std::sin(angle) * (index + 1)));
    if constexpr (Dims == 3)
      radial_points.push_back(Real{0});
  }
  auto mixed_points = make_array(radial_points, {12, static_cast<int>(Dims)});
  auto mixed_faces = dynamic_faces<Index>(
      {0, 3, 7, 12}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  auto mixed =
      tf::cpp::triangulate<Index, Real, Dims>(mixed_faces, mixed_points);
  check_triangle_mesh(mixed, 6, 12);
  require_valid_nondegenerate(mixed);

  auto mesh = dynamic_mesh<Index, Real, Dims>();
  auto mesh_result =
      tf::cpp::triangulate<Index, Real, Dims, tf::dynamic_size>(mesh.mesh());
  check_triangle_mesh(mesh_result, 4, 6);
  CHECK(points_of(mesh_result).data() != points_of(mesh.polygons).data());
}

TEMPLATE_LIST_TEST_CASE(
    "transformed dynamic meshes triangulate exact raw points like Python",
    "[cpp][geometry][triangulate][python-parity][dynamic][transformation]",
    matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  // This quad changes Delaunay diagonal under the x-stretch, so an accidental
  // transformed-point path makes the exact face comparison fail.
  auto raw_points = [&] {
    if constexpr (Dims == 2)
      return points<Real, 2>({{{0, 0}}, {{1, 0}}, {{2, 1}}, {{2, 3}}});
    else
      return points<Real, 3>(
          {{{0, 0, 0}}, {{1, 0, 0}}, {{2, 1, 0}}, {{2, 3, 0}}});
  }();
  auto faces = dynamic_faces<Index>({0, 4}, {0, 1, 2, 3});
  const auto expected =
      tf::cpp::triangulate<Index, Real, Dims>(faces, raw_points);
  mixed_mesh_t<Index, Real, Dims> transformed;
  transformed.polygons =
      tf::cpp::test::polygons_of<Index, Real, Dims, std::vector<Index>,
                                 std::vector<Index>, std::vector<Real>>(
          {0, 4}, {0, 1, 2, 3}, {raw_points.begin(), raw_points.end()});
  transformed.place(x_stretch<Real, Dims>());

  const auto actual = tf::cpp::triangulate<Index, Real, Dims, tf::dynamic_size>(
      transformed.mesh());

  check_equal(faces_of(actual), faces_of(expected));
  check_equal(points_of(actual), raw_points);
  // the placement is the caller's, and a result carries none
  CHECK(transformed.placement.placed);
}

TEMPLATE_LIST_TEST_CASE(
    "triangulation preserves area winding indices and nondegeneracy",
    "[cpp][geometry][triangulate][python-parity][correctness][matrix]",
    matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  auto square = square_points<Real, Dims>();
  auto square_result = tf::cpp::triangulate<Index, Real, Dims>(square);
  CHECK(signed_xy_area(square_result) == Catch::Approx(1.0));
  require_valid_nondegenerate(square_result);

  auto l_shape = [&] {
    if constexpr (Dims == 2)
      return points<Real, 2>(
          {{{0, 0}}, {{2, 0}}, {{2, 1}}, {{1, 1}}, {{1, 2}}, {{0, 2}}});
    else
      return points<Real, 3>({{{0, 0, 0}},
                              {{2, 0, 0}},
                              {{2, 1, 0}},
                              {{1, 1, 0}},
                              {{1, 2, 0}},
                              {{0, 2, 0}}});
  }();
  auto concave = tf::cpp::triangulate<Index, Real, Dims>(l_shape);
  check_triangle_mesh(concave, 4, 6);
  CHECK(signed_xy_area(concave) == Catch::Approx(3.0));
  require_valid_nondegenerate(concave);

  auto large = regular_polygon<Real, Dims>(100);
  auto large_result = tf::cpp::triangulate<Index, Real, Dims>(large);
  check_triangle_mesh(large_result, 98, 100);
  CHECK(std::abs(signed_xy_area(large_result) - std::acos(-1.0)) < 0.01);

  auto clockwise = regular_polygon<Real, Dims>(100, true);
  auto clockwise_result = tf::cpp::triangulate<Index, Real, Dims>(clockwise);
  check_triangle_mesh(clockwise_result, 98, 100);
  CHECK(signed_xy_area(clockwise_result) < 0.0);
  CHECK(std::abs(signed_xy_area(clockwise_result) + std::acos(-1.0)) < 0.01);

  auto hexagon = regular_polygon<Real, Dims>(6);
  auto convex = tf::cpp::triangulate<Index, Real, Dims>(hexagon);
  check_triangle_mesh(convex, 4, 6);
  CHECK(signed_xy_area(convex) ==
        Catch::Approx(3.0 * std::sqrt(3.0) / 2.0).margin(1e-5));
}

TEMPLATE_LIST_TEST_CASE(
    "triangulate validates every carrier and preserves supported empties",
    "[cpp][geometry][triangulate][validation][empty][matrix]", matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  // a default-assembled carrier is the EMPTY mesh, and an empty mesh
  // triangulates to one
  const fixed_mesh_t<Index, Real, Dims> nothing;
  CHECK(tf::cpp::triangulate(nothing.mesh()).faces_buffer().size() == 0);

  auto valid_points = square_points<Real, Dims>();
  auto invalid_points = tf::cpp::nd_array<Real>{};
  auto wrong_points = make_array<Real>({0, 0, 1, 0, 1, 1}, {3, 2});
  if constexpr (Dims == 2)
    wrong_points = make_array<Real>({0, 0, 0, 1, 0, 0, 1, 1, 0}, {3, 3});

  auto rank_one = make_array<Index>({0, 1, 2}, {3});
  auto short_faces = make_array<Index>({0, 1}, {1, 2});
  auto negative = make_array<Index>({0, 1, Index{-1}}, {1, 3});
  auto oversized = make_array<Index>({0, 1, Index{4}}, {1, 3});
  CHECK_THROWS_AS(
      (tf::cpp::triangulate<Index, Real, Dims>(rank_one, valid_points)),
      std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::triangulate<Index, Real, Dims>(short_faces, valid_points)),
      std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::triangulate<Index, Real, Dims>(negative, valid_points)),
      std::out_of_range);
  CHECK_THROWS_AS(
      (tf::cpp::triangulate<Index, Real, Dims>(oversized, valid_points)),
      std::out_of_range);
  CHECK_THROWS_AS((tf::cpp::triangulate<Index, Real, Dims>(
                      make_array<Index>({0, 1, 2, 3}, {1, 4}), wrong_points)),
                  std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::triangulate<Index, Real, Dims>(
                      make_array<Index>({0, 1, 2}, {1, 3}), invalid_points)),
                  std::invalid_argument);

  tf::cpp::offset_blocked_buffer<Index, Index> invalid_dynamic;
  CHECK_THROWS_AS(
      (tf::cpp::triangulate<Index, Real, Dims>(invalid_dynamic, valid_points)),
      std::invalid_argument);
  auto short_dynamic = dynamic_faces<Index>({0, 2}, {0, 1});
  auto negative_dynamic = dynamic_faces<Index>({0, 3}, {0, 1, Index{-1}});
  auto oversized_dynamic = dynamic_faces<Index>({0, 3}, {0, 1, Index{4}});
  CHECK_THROWS_AS(
      (tf::cpp::triangulate<Index, Real, Dims>(short_dynamic, valid_points)),
      std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::triangulate<Index, Real, Dims>(negative_dynamic, valid_points)),
      std::out_of_range);
  CHECK_THROWS_AS((tf::cpp::triangulate<Index, Real, Dims>(oversized_dynamic,
                                                           valid_points)),
                  std::out_of_range);
  auto corrupt_first = dynamic_faces<Index>({0, 4}, {0, 1, 2, 3});
  auto corrupt_final = dynamic_faces<Index>({0, 4}, {0, 1, 2, 3});
  corrupt_first.offsets()[0] = Index{1};
  corrupt_final.offsets()[1] = Index{3};
  CHECK_THROWS_AS(
      (tf::cpp::triangulate<Index, Real, Dims>(corrupt_first, valid_points)),
      std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::triangulate<Index, Real, Dims>(corrupt_final, valid_points)),
      std::invalid_argument);

  CHECK_THROWS_AS((tf::cpp::triangulate<Index, Real, Dims>(invalid_points)),
                  std::invalid_argument);
  auto rank_one_polygon = make_array<Real>({0, 0, 0}, {3});
  auto wrong_polygon = wrong_points;
  auto empty_single = make_array<Real>({}, {0, static_cast<int>(Dims)});
  auto malformed_batch = make_array<Real>({}, {0, 0, static_cast<int>(Dims)});
  CHECK_THROWS_AS((tf::cpp::triangulate<Index, Real, Dims>(rank_one_polygon)),
                  std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::triangulate<Index, Real, Dims>(wrong_polygon)),
                  std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::triangulate<Index, Real, Dims>(empty_single)),
                  std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::triangulate<Index, Real, Dims>(malformed_batch)),
                  std::invalid_argument);

  auto empty_points = make_array<Real>({}, {0, static_cast<int>(Dims)});
  auto empty_triangles = make_array<Index>({}, {0, 3});
  auto empty_quads = make_array<Index>({}, {0, 4});
  auto empty_dynamic = dynamic_faces<Index>({0}, {});
  auto empty_offsets = dynamic_faces<Index>({}, {});
  auto empty_batch = make_array<Real>({}, {0, 4, static_cast<int>(Dims)});
  const fixed_mesh_t<Index, Real, Dims> empty_mesh;
  const mixed_mesh_t<Index, Real, Dims> empty_dynamic_mesh;

  check_triangle_mesh(
      tf::cpp::triangulate<Index, Real, Dims>(empty_mesh.mesh()), 0, 0);
  check_triangle_mesh(tf::cpp::triangulate<Index, Real, Dims, tf::dynamic_size>(
                          empty_dynamic_mesh.mesh()),
                      0, 0);
  check_triangle_mesh(
      tf::cpp::triangulate<Index, Real, Dims>(empty_quads, empty_points), 0, 0);
  check_triangle_mesh(
      tf::cpp::triangulate<Index, Real, Dims>(empty_dynamic, empty_points), 0,
      0);
  check_triangle_mesh(
      tf::cpp::triangulate<Index, Real, Dims>(empty_offsets, empty_points), 0,
      0);
  check_triangle_mesh(tf::cpp::triangulate<Index, Real, Dims>(empty_batch), 0,
                      0);
}

TEMPLATE_LIST_TEST_CASE(
    "generalized futures retain input lifetime through destruction",
    "[cpp][geometry][triangulate][async][ownership][lifetime][matrix]",
    matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;
  using result_type = triangulated_t<Index, Real, Dims>;

  auto mesh = dynamic_mesh<Index, Real, Dims>();
  auto faces = dynamic_faces<Index>({0, 4, 8}, {0, 1, 2, 3, 1, 4, 5, 2});
  auto dynamic_points = shared_quad_points<Real, Dims>();
  auto fixed = make_array<Index>({0, 1, 2, 3}, {1, 4});
  auto fixed_points = square_points<Real, Dims>();
  auto polygon = square_points<Real, Dims>();

  auto mesh_future =
      tf::cpp::async::triangulate<Index, Real, Dims, tf::dynamic_size>(
          mesh.mesh());
  auto dynamic_future =
      tf::cpp::async::triangulate<Index, Real, Dims>(faces, dynamic_points);
  auto fixed_future =
      tf::cpp::async::triangulate<Index, Real, Dims>(fixed, fixed_points);
  int submissions = 0;
  auto polygon_future = tf::cpp::async::triangulate<Index, Real, Dims>(
      counting_resolver{&submissions}, polygon);

  STATIC_REQUIRE(
      std::is_same_v<decltype(mesh_future), std::future<result_type>>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(dynamic_future), std::future<result_type>>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(fixed_future), std::future<result_type>>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(polygon_future), std::future<result_type>>);
  CHECK(submissions == 1);

  // Destruction is the array-overload ownership guarantee. The handles are not
  // mutated here: shallow ownership deliberately does not promise isolation
  // from writes through aliases. A mesh is BORROWED, so its arrays stay.
  faces.destroy();
  dynamic_points.destroy();
  fixed.destroy();
  fixed_points.destroy();
  polygon.destroy();

  check_triangle_mesh(mesh_future.get(), 4, 6);
  check_triangle_mesh(dynamic_future.get(), 4, 6);
  check_triangle_mesh(fixed_future.get(), 2, 4);
  check_triangle_mesh(polygon_future.get(), 2, 4);
}

TEST_CASE("array resolver overloads retain aliases for lifetime only",
          "[cpp][geometry][triangulate][async][ownership][lifetime-only]") {
  using Real = double;
  using Index = std::int64_t;
  constexpr std::size_t Dims = 3;
  const auto submissions = std::make_shared<std::atomic<int>>(0);

  auto dynamic = dynamic_faces<Index>({0, 3}, {0, 1, 2});
  auto dynamic_data_alias = dynamic.data();
  auto dynamic_points =
      points<Real, Dims>({{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}, {{2, 1, 0}}});
  auto dynamic_pending = tf::cpp::async::triangulate<Index, Real, Dims>(
      mutating_resolver{submissions, [&] { dynamic_data_alias[2] = Index{3}; }},
      dynamic, dynamic_points);
  auto dynamic_output = dynamic_pending.get();
  const auto &dynamic_output_faces = faces_of(dynamic_output);
  CHECK(std::find(dynamic_output_faces.begin(), dynamic_output_faces.end(),
                  Index{3}) != dynamic_output_faces.end());

  auto fixed = make_array<Index>({0, 1, 2}, {1, 3});
  auto fixed_points = dynamic_points.deep_copy();
  auto fixed_pending = tf::cpp::async::triangulate<Index, Real, Dims>(
      mutating_resolver{submissions, [&] { fixed[2] = Index{3}; }}, fixed,
      fixed_points);
  auto fixed_output = fixed_pending.get();
  CHECK(faces_of(fixed_output)[2] == Index{3});

  auto polygon = square_points<Real, Dims>();
  auto polygon_pending = tf::cpp::async::triangulate<Index, Real, Dims>(
      mutating_resolver{submissions, [&] { polygon[0] = Real{-4}; }}, polygon);
  auto polygon_output = polygon_pending.get();
  CHECK(points_of(polygon_output)[0] == Real{-4});

  auto legacy_fixed = make_array<std::int32_t>({0, 1, 2}, {1, 3});
  auto legacy_points = dynamic_points.deep_copy();
  auto legacy_pending = tf::cpp::async::triangulate<std::int32_t, Real, 3>(
      mutating_resolver{submissions,
                        [&] { legacy_fixed[2] = std::int32_t{3}; }},
      legacy_fixed, legacy_points);
  auto legacy_output = legacy_pending.get();
  const auto &legacy_output_faces = faces_of(legacy_output);
  CHECK(std::find(legacy_output_faces.begin(), legacy_output_faces.end(),
                  std::int32_t{3}) != legacy_output_faces.end());
  CHECK(submissions->load(std::memory_order_relaxed) == 4);
}

TEST_CASE("async mesh triangulation carries the reading it was handed",
          "[cpp][geometry][triangulate][async][reading]") {
  using Real = double;
  using Index = std::int64_t;
  constexpr std::size_t Dims = 3;

  const auto input = dynamic_mesh<Index, Real, Dims>();
  const auto expected =
      tf::cpp::triangulate<Index, Real, Dims, tf::dynamic_size>(input.mesh());
  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto pending =
      tf::cpp::async::triangulate<Index, Real, Dims, tf::dynamic_size>(
          mutating_resolver{submissions, [] {}}, input.mesh());

  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  const auto output = pending.get();
  CHECK(output.faces_buffer().size() == expected.faces_buffer().size());
  CHECK(output.points_buffer().size() == expected.points_buffer().size());
  CHECK(signed_xy_area(output) == Catch::Approx(signed_xy_area(expected)));
  check_equal(points_of(output), points_of(expected));
}

TEMPLATE_TEST_CASE("legacy indexed and mesh overloads own their point table",
                   "[cpp][geometry][triangulate][legacy][ownership]", float,
                   double) {
  using Real = TestType;
  using Index = std::int32_t;

  auto dynamic = dynamic_faces<Index>({0, 4}, {0, 1, 2, 3});
  auto dynamic_points = square_points<Real, 3>();
  const auto *dynamic_storage = dynamic_points.raw_data();
  auto dynamic_output =
      tf::cpp::triangulate<Index, Real>(dynamic, dynamic_points);
  CHECK(static_cast<const void *>(points_of(dynamic_output).data()) !=
        static_cast<const void *>(dynamic_storage));
  check_equal(points_of(dynamic_output), dynamic_points);
  dynamic_points[0] = Real{-2};
  CHECK(points_of(dynamic_output)[0] != Real{-2});

  auto fixed = make_array<Index>({0, 1, 2, 3}, {1, 4});
  auto fixed_points = square_points<Real, 3>();
  const auto *fixed_storage = fixed_points.raw_data();
  auto fixed_output = tf::cpp::triangulate<Index, Real>(fixed, fixed_points);
  CHECK(static_cast<const void *>(points_of(fixed_output).data()) !=
        static_cast<const void *>(fixed_storage));
  fixed_output.points_buffer().data_buffer()[1] = Real{7};
  CHECK(fixed_points[1] != Real{7});

  auto mesh_points = points<Real, 3>({{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}});
  fixed_mesh_t<tf::cpp::default_index_t, Real, 3> mesh;
  mesh.polygons =
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real, 3,
                                 std::vector<tf::cpp::default_index_t>,
                                 std::vector<Real>>(
          {0, 1, 2}, {mesh_points.begin(), mesh_points.end()});
  auto mesh_output =
      tf::cpp::triangulate<tf::cpp::default_index_t, Real>(mesh.mesh());
  CHECK(points_of(mesh_output).data() != points_of(mesh.polygons).data());
  check_equal(points_of(mesh_output), mesh_points);

  auto polygon = square_points<Real, 3>();
  const auto *polygon_storage = polygon.raw_data();
  auto polygon_output =
      tf::cpp::triangulate<tf::cpp::default_index_t, Real>(polygon);
  CHECK(static_cast<const void *>(points_of(polygon_output).data()) !=
        static_cast<const void *>(polygon_storage));
}

TEST_CASE("the triangulate symbols link from the native archive",
          "[cpp][geometry][triangulate][abi][archive-link]") {
  using Real = float;
  using Index = std::int32_t;
  using result_type = triangulated_t<tf::cpp::default_index_t, Real, 3>;
  using mesh_type = tf::cpp::mesh<tf::cpp::default_index_t, Real, 3, 3>;
  using dynamic_type =
      tf::cpp::offset_blocked_buffer<std::int32_t, std::int32_t>;
  using mesh_fn = result_type (*)(const mesh_type &);
  using dynamic_fn =
      result_type (*)(const dynamic_type &, const tf::cpp::nd_array<Real> &);
  using fixed_fn = result_type (*)(const tf::cpp::nd_array<Index> &,
                                   const tf::cpp::nd_array<Real> &);
  using polygon_fn = result_type (*)(const tf::cpp::nd_array<Real> &);

  const mesh_fn mesh_symbol = &tf::cpp::triangulate<Index, Real, 3, 3>;
  const dynamic_fn dynamic_symbol = &tf::cpp::triangulate<Index, Real, 3>;
  const fixed_fn fixed_symbol = &tf::cpp::triangulate<Index, Real, 3>;
  const polygon_fn polygon_symbol = &tf::cpp::triangulate<Index, Real, 3>;

  CHECK(mesh_symbol != nullptr);
  CHECK(dynamic_symbol != nullptr);
  CHECK(fixed_symbol != nullptr);
  CHECK(polygon_symbol != nullptr);

  auto polygon = square_points<Real, 3>();
  auto faces = make_array<Index>({0, 1, 2, 3}, {1, 4});
  auto dynamic = dynamic_faces<Index>({0, 4}, {0, 1, 2, 3});
  const auto mesh_points =
      points<Real, 3>({{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}}});
  fixed_mesh_t<tf::cpp::default_index_t, Real, 3> mesh;
  mesh.polygons =
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real, 3,
                                 std::vector<tf::cpp::default_index_t>,
                                 std::vector<Real>>(
          {0, 1, 2}, {mesh_points.begin(), mesh_points.end()});
  CHECK(polygon_symbol(polygon).faces_buffer().size() == 2);
  CHECK(fixed_symbol(faces, polygon).faces_buffer().size() == 2);
  CHECK(dynamic_symbol(dynamic, polygon).faces_buffer().size() == 2);
  CHECK(mesh_symbol(mesh.mesh()).faces_buffer().size() == 1);
}

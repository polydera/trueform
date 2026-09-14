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
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/geometry/async/make_box_mesh.hpp"
#include "trueform/cpp/geometry/async/make_cylinder_mesh.hpp"
#include "trueform/cpp/geometry/async/make_plane_mesh.hpp"
#include "trueform/cpp/geometry/async/make_sphere_mesh.hpp"
#include "trueform/cpp/geometry/async/make_tube_mesh.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/geometry/make_cylinder_mesh.hpp"
#include "trueform/cpp/geometry/make_plane_mesh.hpp"
#include "trueform/cpp/geometry/make_sphere_mesh.hpp"
#include "trueform/cpp/geometry/make_tube_mesh.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

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

template <typename RealT, typename IndexT> struct mesh_primitive_case {
  using real_type = RealT;
  using index_type = IndexT;
};

using mesh_primitive_matrix =
    std::tuple<mesh_primitive_case<float, std::int32_t>,
               mesh_primitive_case<float, std::int64_t>,
               mesh_primitive_case<double, std::int32_t>,
               mesh_primitive_case<double, std::int64_t>>;

template <typename Index, typename Real>
using primitive_mesh = tf::polygons_buffer<Index, Real, 3, 3>;

/// A primitive is core's own storage, so a check reads its flat arrays where
/// they lie.
template <typename Index, typename Real>
auto points_of(const primitive_mesh<Index, Real> &value)
    -> const tf::buffer<Real> & {
  return value.points_buffer().data_buffer();
}

template <typename Index, typename Real>
auto faces_of(const primitive_mesh<Index, Real> &value)
    -> const tf::buffer<Index> & {
  return value.faces_buffer().data_buffer();
}

template <typename Index, typename Real>
auto point_count(const primitive_mesh<Index, Real> &value) -> int {
  return static_cast<int>(value.points_buffer().size());
}

template <typename Index, typename Real>
auto face_count(const primitive_mesh<Index, Real> &value) -> int {
  return static_cast<int>(value.faces_buffer().size());
}

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

/// The two arrays a polyline is, which is what the tube entry takes.
template <typename Real> struct tube_input {
  tf::cpp::offset_blocked_buffer<tf::cpp::default_index_t,
                                 tf::cpp::default_index_t>
      paths;
  tf::cpp::nd_array<Real> points;
};

template <typename Real> auto straight_curve() -> tube_input<Real> {
  return {tf::cpp::offset_blocked_buffer<std::int32_t, std::int32_t>::create(
              make_array<std::int32_t>({0, 3}, {2}),
              make_array<std::int32_t>({0, 1, 2}, {3})),
          make_array<Real>({Real{0}, Real{0}, Real{0}, Real{1}, Real{0},
                            Real{0}, Real{2}, Real{0}, Real{0}},
                           {3, 3})};
}

template <typename Real> auto empty_curves() -> tube_input<Real> {
  return {tf::cpp::offset_blocked_buffer<std::int32_t, std::int32_t>::create(
              make_array<std::int32_t>({0}, {1}),
              make_array<std::int32_t>({}, {0})),
          make_array<Real>({}, {0, 3})};
}

template <typename Real> auto singleton_curve() -> tube_input<Real> {
  return {tf::cpp::offset_blocked_buffer<std::int32_t, std::int32_t>::create(
              make_array<std::int32_t>({0, 1}, {2}),
              make_array<std::int32_t>({0}, {1})),
          make_array<Real>({Real{0}, Real{0}, Real{0}}, {1, 3})};
}

template <typename Index, typename Real>
auto coordinate_min(const primitive_mesh<Index, Real> &value, int coordinate)
    -> Real {
  const auto &points = points_of(value);
  auto result = points[static_cast<std::size_t>(coordinate)];
  for (int point = 1; point < point_count(value); ++point)
    result = std::min(result,
                      points[static_cast<std::size_t>(3 * point + coordinate)]);
  return result;
}

template <typename Index, typename Real>
auto coordinate_max(const primitive_mesh<Index, Real> &value, int coordinate)
    -> Real {
  const auto &points = points_of(value);
  auto result = points[static_cast<std::size_t>(coordinate)];
  for (int point = 1; point < point_count(value); ++point)
    result = std::max(result,
                      points[static_cast<std::size_t>(3 * point + coordinate)]);
  return result;
}

template <typename Real> auto tolerance() -> double {
  return std::is_same_v<Real, float> ? 1e-5 : 1e-12;
}

template <typename Index, typename Real>
auto check_mesh_shape(const primitive_mesh<Index, Real> &value, int points,
                      int faces) -> void {
  CHECK(point_count(value) == points);
  CHECK(face_count(value) == faces);
  CHECK(points_of(value).size() == static_cast<std::size_t>(points) * 3);
  CHECK(faces_of(value).size() == static_cast<std::size_t>(faces) * 3);
}

template <typename Index, typename Real>
auto triangle_cross(const primitive_mesh<Index, Real> &value, int face)
    -> std::array<double, 3> {
  const auto &faces = faces_of(value);
  const auto &points = points_of(value);
  const auto face_offset = static_cast<std::size_t>(3 * face);
  const auto first = static_cast<std::size_t>(3 * faces[face_offset]);
  const auto second = static_cast<std::size_t>(3 * faces[face_offset + 1]);
  const auto third = static_cast<std::size_t>(3 * faces[face_offset + 2]);
  const double ab[]{
      static_cast<double>(points[second] - points[first]),
      static_cast<double>(points[second + 1] - points[first + 1]),
      static_cast<double>(points[second + 2] - points[first + 2])};
  const double ac[]{static_cast<double>(points[third] - points[first]),
                    static_cast<double>(points[third + 1] - points[first + 1]),
                    static_cast<double>(points[third + 2] - points[first + 2])};
  return {ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2],
          ab[0] * ac[1] - ab[1] * ac[0]};
}

template <typename Index, typename Real>
auto check_triangle_topology(const primitive_mesh<Index, Real> &value,
                             int expected_boundary_edges) -> void {
  struct edge_incidence {
    int count = 0;
    int direction_balance = 0;
  };

  std::map<std::pair<Index, Index>, edge_incidence> edge_counts;
  std::vector<bool> referenced(static_cast<std::size_t>(point_count(value)),
                               false);
  const auto &faces = faces_of(value);
  const auto &points = points_of(value);
  for (std::size_t coordinate = 0; coordinate < points.size(); ++coordinate)
    CHECK(std::isfinite(static_cast<double>(points[coordinate])));

  for (int face = 0; face < face_count(value); ++face) {
    const auto offset = static_cast<std::size_t>(3 * face);
    const Index vertices[]{faces[offset], faces[offset + 1], faces[offset + 2]};
    REQUIRE(vertices[0] != vertices[1]);
    REQUIRE(vertices[1] != vertices[2]);
    REQUIRE(vertices[2] != vertices[0]);
    for (const auto vertex : vertices) {
      REQUIRE(vertex >= Index{0});
      REQUIRE(vertex < static_cast<Index>(point_count(value)));
      referenced[static_cast<std::size_t>(vertex)] = true;
    }

    const auto cross = triangle_cross(value, face);
    const auto squared_area_times_four =
        cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
    CHECK(squared_area_times_four > 0.0);

    for (int edge = 0; edge < 3; ++edge) {
      const auto first = vertices[edge];
      const auto second = vertices[(edge + 1) % 3];
      const auto canonical = std::minmax(first, second);
      auto &incidence = edge_counts[canonical];
      ++incidence.count;
      incidence.direction_balance += first < second ? 1 : -1;
    }
  }

  for (const auto is_referenced : referenced)
    CHECK(is_referenced);

  int boundary_edges = 0;
  for (const auto &entry : edge_counts) {
    CHECK((entry.second.count == 1 || entry.second.count == 2));
    if (entry.second.count == 1) {
      ++boundary_edges;
      CHECK(std::abs(entry.second.direction_balance) == 1);
    } else {
      CHECK(entry.second.direction_balance == 0);
    }
  }
  CHECK(boundary_edges == expected_boundary_edges);
}

template <typename Index, typename Real>
auto signed_volume(const primitive_mesh<Index, Real> &value) -> double {
  const auto &faces = faces_of(value);
  const auto &points = points_of(value);
  double result = 0.0;
  for (int face = 0; face < face_count(value); ++face) {
    const auto face_offset = static_cast<std::size_t>(3 * face);
    const auto first = static_cast<std::size_t>(3 * faces[face_offset]);
    const auto second = static_cast<std::size_t>(3 * faces[face_offset + 1]);
    const auto third = static_cast<std::size_t>(3 * faces[face_offset + 2]);
    const double cross[]{
        static_cast<double>(points[second + 1]) * points[third + 2] -
            static_cast<double>(points[second + 2]) * points[third + 1],
        static_cast<double>(points[second + 2]) * points[third] -
            static_cast<double>(points[second]) * points[third + 2],
        static_cast<double>(points[second]) * points[third + 1] -
            static_cast<double>(points[second + 1]) * points[third]};
    result += (static_cast<double>(points[first]) * cross[0] +
               static_cast<double>(points[first + 1]) * cross[1] +
               static_cast<double>(points[first + 2]) * cross[2]) /
              6.0;
  }
  return result;
}

template <typename Index, typename Real>
auto projected_xy_area(const primitive_mesh<Index, Real> &value) -> double {
  double result = 0.0;
  for (int face = 0; face < face_count(value); ++face) {
    const auto cross = triangle_cross(value, face);
    CHECK(cross[2] > 0.0);
    result += cross[2] / 2.0;
  }
  return result;
}

template <typename Index, typename Real>
auto check_axis_bounds(const primitive_mesh<Index, Real> &value, int coordinate,
                       double expected_minimum, double expected_maximum)
    -> void {
  const auto margin =
      10.0 * tolerance<Real>() *
      std::max({1.0, std::abs(expected_minimum), std::abs(expected_maximum)});
  CHECK(static_cast<double>(coordinate_min(value, coordinate)) ==
        Catch::Approx(expected_minimum).margin(margin));
  CHECK(static_cast<double>(coordinate_max(value, coordinate)) ==
        Catch::Approx(expected_maximum).margin(margin));
}

template <typename Index, typename Real>
auto check_sphere(const primitive_mesh<Index, Real> &value, Real radius,
                  int stacks, int segments) -> void {
  check_mesh_shape(value, 2 + (stacks - 1) * segments,
                   2 * (stacks - 1) * segments);
  check_triangle_topology(value, 0);
  const auto &points = points_of(value);
  const auto expected_radius = static_cast<double>(radius);
  const auto margin =
      10.0 * tolerance<Real>() * std::max(1.0, std::abs(expected_radius));
  for (int point = 0; point < point_count(value); ++point) {
    const auto offset = static_cast<std::size_t>(3 * point);
    const auto distance =
        std::sqrt(static_cast<double>(points[offset]) * points[offset] +
                  static_cast<double>(points[offset + 1]) * points[offset + 1] +
                  static_cast<double>(points[offset + 2]) * points[offset + 2]);
    CHECK(distance == Catch::Approx(expected_radius).margin(margin));
  }
  check_axis_bounds(value, 2, -expected_radius, expected_radius);
}

template <typename Index, typename Real>
auto check_cylinder(const primitive_mesh<Index, Real> &value, Real radius,
                    Real height, int segments) -> void {
  check_mesh_shape(value, 2 + 2 * segments, 4 * segments);
  check_triangle_topology(value, 0);
  const auto &points = points_of(value);
  const auto expected_radius = static_cast<double>(radius);
  const auto margin =
      10.0 * tolerance<Real>() * std::max(1.0, std::abs(expected_radius));
  int centers = 0;
  for (int point = 0; point < point_count(value); ++point) {
    const auto offset = static_cast<std::size_t>(3 * point);
    const auto radial_distance =
        std::sqrt(static_cast<double>(points[offset]) * points[offset] +
                  static_cast<double>(points[offset + 1]) * points[offset + 1]);
    if (radial_distance <= margin) {
      ++centers;
      CHECK(radial_distance == Catch::Approx(0.0).margin(margin));
    } else {
      CHECK(radial_distance == Catch::Approx(expected_radius).margin(margin));
    }
  }
  CHECK(centers == 2);
  check_axis_bounds(value, 0, -expected_radius, expected_radius);
  check_axis_bounds(value, 1, -expected_radius, expected_radius);
  check_axis_bounds(value, 2, -static_cast<double>(height) / 2.0,
                    static_cast<double>(height) / 2.0);
}

template <typename Index, typename Real>
auto check_box(const primitive_mesh<Index, Real> &value, Real width,
               Real height, Real depth, int width_ticks, int height_ticks,
               int depth_ticks) -> void {
  const auto face_cells = width_ticks * height_ticks +
                          width_ticks * depth_ticks +
                          height_ticks * depth_ticks;
  check_mesh_shape(value, 2 * face_cells + 2, 4 * face_cells);
  check_triangle_topology(value, 0);
  check_axis_bounds(value, 0, -static_cast<double>(width) / 2.0,
                    static_cast<double>(width) / 2.0);
  check_axis_bounds(value, 1, -static_cast<double>(height) / 2.0,
                    static_cast<double>(height) / 2.0);
  check_axis_bounds(value, 2, -static_cast<double>(depth) / 2.0,
                    static_cast<double>(depth) / 2.0);
  const auto expected_volume = static_cast<double>(width) * height * depth;
  CHECK(signed_volume(value) ==
        Catch::Approx(expected_volume)
            .margin(10.0 * tolerance<Real>() *
                    std::max(1.0, std::abs(expected_volume))));
}

template <typename Index, typename Real>
auto check_plane(const primitive_mesh<Index, Real> &value, Real width,
                 Real height, int width_ticks, int height_ticks) -> void {
  check_mesh_shape(value, (width_ticks + 1) * (height_ticks + 1),
                   2 * width_ticks * height_ticks);
  check_triangle_topology(value, 2 * (width_ticks + height_ticks));
  check_axis_bounds(value, 0, -static_cast<double>(width) / 2.0,
                    static_cast<double>(width) / 2.0);
  check_axis_bounds(value, 1, -static_cast<double>(height) / 2.0,
                    static_cast<double>(height) / 2.0);
  check_axis_bounds(value, 2, 0.0, 0.0);
  const auto expected_area = static_cast<double>(width) * height;
  CHECK(projected_xy_area(value) ==
        Catch::Approx(expected_area)
            .margin(10.0 * tolerance<Real>() *
                    std::max(1.0, std::abs(expected_area))));
}

} // namespace

TEMPLATE_LIST_TEST_CASE(
    "typed sphere factories cover every Python parameter row analytically",
    "[cpp][geometry][mesh-primitives][python-parity][typed-index][sphere]",
    mesh_primitive_matrix) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using mesh_type = primitive_mesh<Index, Real>;

  auto dtype_case = tf::cpp::make_sphere_mesh<Index, Real>(Real{1}, 10, 10);
  STATIC_REQUIRE(std::is_same_v<decltype(dtype_case), mesh_type>);
  check_sphere(dtype_case, Real{1}, 10, 10);

  auto radius_case = tf::cpp::make_sphere_mesh<Index, Real>(Real{3.5}, 10, 10);
  check_sphere(radius_case, Real{3.5}, 10, 10);

  for (const auto [stacks, segments] :
       std::array{std::pair{5, 5}, std::pair{10, 10}, std::pair{20, 20},
                  std::pair{10, 30}}) {
    CAPTURE(stacks, segments);
    auto value =
        tf::cpp::make_sphere_mesh<Index, Real>(Real{1}, stacks, segments);
    check_sphere(value, Real{1}, stacks, segments);
  }

  for (const auto radius : std::array{Real{0.5}, Real{1}, Real{2}, Real{5}}) {
    CAPTURE(radius);
    auto value = tf::cpp::make_sphere_mesh<Index, Real>(radius, 40, 40);
    check_sphere(value, radius, 40, 40);
    const auto pi = std::acos(-1.0);
    const auto expected_volume =
        4.0 * pi * static_cast<double>(radius) * radius * radius / 3.0;
    CHECK(signed_volume(value) == Catch::Approx(expected_volume).epsilon(0.01));
  }
}

TEMPLATE_LIST_TEST_CASE(
    "typed cylinder factories cover every Python parameter row analytically",
    "[cpp][geometry][mesh-primitives][python-parity][typed-index][cylinder]",
    mesh_primitive_matrix) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using mesh_type = primitive_mesh<Index, Real>;

  auto dtype_case =
      tf::cpp::make_cylinder_mesh<Index, Real>(Real{1}, Real{2}, 16);
  STATIC_REQUIRE(std::is_same_v<decltype(dtype_case), mesh_type>);
  check_cylinder(dtype_case, Real{1}, Real{2}, 16);

  auto bounds_case =
      tf::cpp::make_cylinder_mesh<Index, Real>(Real{2}, Real{4}, 32);
  check_cylinder(bounds_case, Real{2}, Real{4}, 32);

  for (const auto segments : std::array{8, 16, 32, 64}) {
    CAPTURE(segments);
    auto value =
        tf::cpp::make_cylinder_mesh<Index, Real>(Real{1}, Real{2}, segments);
    check_cylinder(value, Real{1}, Real{2}, segments);
  }

  for (const auto [radius, height] :
       std::array{std::pair{Real{0.5}, Real{1}}, std::pair{Real{1}, Real{2}},
                  std::pair{Real{2}, Real{3}}}) {
    CAPTURE(radius, height);
    auto value = tf::cpp::make_cylinder_mesh<Index, Real>(radius, height, 64);
    check_cylinder(value, radius, height, 64);
    const auto expected_volume =
        std::acos(-1.0) * static_cast<double>(radius) * radius * height;
    CHECK(signed_volume(value) == Catch::Approx(expected_volume).epsilon(0.01));
  }
}

TEMPLATE_LIST_TEST_CASE(
    "typed box factories cover every Python parameter row analytically",
    "[cpp][geometry][mesh-primitives][python-parity][typed-index][box]",
    mesh_primitive_matrix) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using mesh_type = primitive_mesh<Index, Real>;

  auto dtype_case =
      tf::cpp::make_box_mesh<Index, Real>(Real{2}, Real{1}, Real{3});
  STATIC_REQUIRE(std::is_same_v<decltype(dtype_case), mesh_type>);
  check_box(dtype_case, Real{2}, Real{1}, Real{3}, 1, 1, 1);

  auto bounds_case =
      tf::cpp::make_box_mesh<Index, Real>(Real{4}, Real{2}, Real{6});
  check_box(bounds_case, Real{4}, Real{2}, Real{6}, 1, 1, 1);

  for (const auto [width, height, depth] :
       std::array{std::tuple{Real{1}, Real{1}, Real{1}},
                  std::tuple{Real{2}, Real{3}, Real{4}},
                  std::tuple{Real{0.5}, Real{1.5}, Real{2.5}}}) {
    CAPTURE(width, height, depth);
    auto value = tf::cpp::make_box_mesh<Index, Real>(width, height, depth);
    check_box(value, width, height, depth, 1, 1, 1);
  }

  for (const auto [width_ticks, height_ticks, depth_ticks] :
       std::array{std::tuple{1, 1, 1}, std::tuple{2, 2, 2}, std::tuple{3, 1, 1},
                  std::tuple{1, 4, 1}, std::tuple{1, 1, 5}, std::tuple{2, 3, 4},
                  std::tuple{5, 5, 5}}) {
    CAPTURE(width_ticks, height_ticks, depth_ticks);
    auto value = tf::cpp::make_box_mesh<Index, Real>(
        Real{2}, Real{3}, Real{4}, width_ticks, height_ticks, depth_ticks);
    check_box(value, Real{2}, Real{3}, Real{4}, width_ticks, height_ticks,
              depth_ticks);
  }

  for (const auto [width_ticks, height_ticks, depth_ticks] : std::array{
           std::tuple{1, 1, 1}, std::tuple{2, 2, 2}, std::tuple{3, 4, 5}}) {
    CAPTURE(width_ticks, height_ticks, depth_ticks);
    auto value = tf::cpp::make_box_mesh<Index, Real>(
        Real{2}, Real{3}, Real{4}, width_ticks, height_ticks, depth_ticks);
    check_box(value, Real{2}, Real{3}, Real{4}, width_ticks, height_ticks,
              depth_ticks);
  }

  for (const auto [width_ticks, height_ticks, depth_ticks] :
       std::array{std::tuple{2, 2, 2}, std::tuple{3, 3, 3}}) {
    CAPTURE(width_ticks, height_ticks, depth_ticks);
    auto value = tf::cpp::make_box_mesh<Index, Real>(
        Real{4}, Real{2}, Real{6}, width_ticks, height_ticks, depth_ticks);
    check_box(value, Real{4}, Real{2}, Real{6}, width_ticks, height_ticks,
              depth_ticks);
  }
}

TEMPLATE_LIST_TEST_CASE(
    "typed plane factories cover every Python parameter row analytically",
    "[cpp][geometry][mesh-primitives][python-parity][typed-index][plane]",
    mesh_primitive_matrix) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using mesh_type = primitive_mesh<Index, Real>;

  auto dtype_case = tf::cpp::make_plane_mesh<Index, Real>(Real{10}, Real{5});
  STATIC_REQUIRE(std::is_same_v<decltype(dtype_case), mesh_type>);
  check_plane(dtype_case, Real{10}, Real{5}, 1, 1);

  auto bounds_case = tf::cpp::make_plane_mesh<Index, Real>(Real{8}, Real{4});
  check_plane(bounds_case, Real{8}, Real{4}, 1, 1);

  for (const auto [width, height] :
       std::array{std::pair{Real{1}, Real{1}}, std::pair{Real{4}, Real{3}},
                  std::pair{Real{10}, Real{5}}}) {
    CAPTURE(width, height);
    auto value = tf::cpp::make_plane_mesh<Index, Real>(width, height);
    check_plane(value, width, height, 1, 1);
  }

  for (const auto [width_ticks, height_ticks] :
       std::array{std::pair{1, 1}, std::pair{2, 2}, std::pair{5, 3},
                  std::pair{10, 10}}) {
    CAPTURE(width_ticks, height_ticks);
    auto value = tf::cpp::make_plane_mesh<Index, Real>(
        Real{6}, Real{4}, width_ticks, height_ticks);
    check_plane(value, Real{6}, Real{4}, width_ticks, height_ticks);
  }

  for (const auto [width_ticks, height_ticks] :
       std::array{std::pair{1, 1}, std::pair{5, 3}, std::pair{10, 10}}) {
    CAPTURE(width_ticks, height_ticks);
    auto value = tf::cpp::make_plane_mesh<Index, Real>(
        Real{4}, Real{3}, width_ticks, height_ticks);
    check_plane(value, Real{4}, Real{3}, width_ticks, height_ticks);
  }

  for (const auto [width_ticks, height_ticks] :
       std::array{std::pair{1, 1}, std::pair{3, 3}, std::pair{5, 5}}) {
    CAPTURE(width_ticks, height_ticks);
    auto value = tf::cpp::make_plane_mesh<Index, Real>(
        Real{4}, Real{3}, width_ticks, height_ticks);
    check_plane(value, Real{4}, Real{3}, width_ticks, height_ticks);
  }
}

TEMPLATE_LIST_TEST_CASE(
    "typed mesh primitive validation is independent of index storage",
    "[cpp][geometry][mesh-primitives][validation][typed-index]",
    mesh_primitive_matrix) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;

  CHECK_THROWS_AS((tf::cpp::make_sphere_mesh<Index, Real>(Real{1}, 1, 3)),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::make_cylinder_mesh<Index, Real>(Real{1}, Real{1}, 2)),
      std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::make_box_mesh<Index, Real>(Real{1}, Real{1}, Real{1}, 0, 1, 1)),
      std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::make_plane_mesh<Index, Real>(Real{1}, Real{1}, 1, 0)),
      std::invalid_argument);
}

TEMPLATE_LIST_TEST_CASE(
    "typed mesh primitive futures and resolvers own exact typed results",
    "[cpp][geometry][mesh-primitives][async][ownership][typed-index]",
    mesh_primitive_matrix) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using future_type = std::future<primitive_mesh<Index, Real>>;

  auto sphere = tf::cpp::async::make_sphere_mesh<Index, Real>(Real{2}, 4, 6);
  auto cylinder =
      tf::cpp::async::make_cylinder_mesh<Index, Real>(Real{2}, Real{6}, 8);
  auto box =
      tf::cpp::async::make_box_mesh<Index, Real>(Real{2}, Real{4}, Real{6});
  auto subdivided = tf::cpp::async::make_box_mesh<Index, Real>(
      Real{2}, Real{4}, Real{6}, 2, 3, 4);
  auto plane = tf::cpp::async::make_plane_mesh<Index, Real>(Real{8}, Real{6});
  STATIC_REQUIRE(std::is_same_v<decltype(sphere), future_type>);
  STATIC_REQUIRE(std::is_same_v<decltype(cylinder), future_type>);
  STATIC_REQUIRE(std::is_same_v<decltype(box), future_type>);
  STATIC_REQUIRE(std::is_same_v<decltype(subdivided), future_type>);
  STATIC_REQUIRE(std::is_same_v<decltype(plane), future_type>);

  check_mesh_shape(sphere.get(), 20, 36);
  check_mesh_shape(cylinder.get(), 18, 32);
  check_mesh_shape(box.get(), 8, 12);
  check_mesh_shape(subdivided.get(), 54, 104);
  check_mesh_shape(plane.get(), 4, 2);

  int submissions = 0;
  const auto resolver = counting_resolver{&submissions};
  auto resolved_sphere =
      tf::cpp::async::make_sphere_mesh<Index, Real>(resolver, Real{2}, 4, 6);
  auto resolved_cylinder = tf::cpp::async::make_cylinder_mesh<Index, Real>(
      resolver, Real{2}, Real{6}, 8);
  auto resolved_box = tf::cpp::async::make_box_mesh<Index, Real>(
      resolver, Real{2}, Real{4}, Real{6});
  auto resolved_subdivided = tf::cpp::async::make_box_mesh<Index, Real>(
      resolver, Real{2}, Real{4}, Real{6}, 2, 3, 4);
  auto resolved_plane =
      tf::cpp::async::make_plane_mesh<Index, Real>(resolver, Real{8}, Real{6});
  STATIC_REQUIRE(std::is_same_v<decltype(resolved_sphere), future_type>);
  STATIC_REQUIRE(std::is_same_v<decltype(resolved_cylinder), future_type>);
  STATIC_REQUIRE(std::is_same_v<decltype(resolved_box), future_type>);
  STATIC_REQUIRE(std::is_same_v<decltype(resolved_subdivided), future_type>);
  STATIC_REQUIRE(std::is_same_v<decltype(resolved_plane), future_type>);
  CHECK(submissions == 5);

  auto owned_sphere = resolved_sphere.get();
  check_mesh_shape(owned_sphere, 20, 36);
  check_mesh_shape(resolved_cylinder.get(), 18, 32);
  check_mesh_shape(resolved_box.get(), 8, 12);
  check_mesh_shape(resolved_subdivided.get(), 54, 104);
  check_mesh_shape(resolved_plane.get(), 4, 2);

  // a primitive states storage of its own, so what a caller keeps outlives
  // every handle the call was made through
  const auto first_face = faces_of(owned_sphere)[0];
  const auto first_point = points_of(owned_sphere)[0];
  auto kept = std::move(owned_sphere);
  CHECK(faces_of(kept)[0] == first_face);
  CHECK(points_of(kept)[0] == first_point);

  auto failure = tf::cpp::async::make_sphere_mesh<Index, Real>(Real{1}, 1, 3);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);
}

TEMPLATE_TEST_CASE("async mesh primitives mirror synchronous operations",
                   "[cpp][geometry][mesh-primitives][async]", float, double) {
  auto sphere = tf::cpp::async::make_sphere_mesh(TestType{2}, 4, 6);
  auto cylinder =
      tf::cpp::async::make_cylinder_mesh(TestType{2}, TestType{6}, 8);
  auto box =
      tf::cpp::async::make_box_mesh(TestType{2}, TestType{4}, TestType{6});
  auto subdivided = tf::cpp::async::make_box_mesh(TestType{2}, TestType{4},
                                                  TestType{6}, 2, 3, 4);
  int submissions = 0;
  auto plane = tf::cpp::async::make_plane_mesh(counting_resolver{&submissions},
                                               TestType{8}, TestType{6}, 4, 3);

  static_assert(
      std::is_same_v<decltype(sphere),
                     std::future<tf::polygons_buffer<tf::cpp::default_index_t,
                                                     TestType, 3, 3>>>);
  static_assert(
      std::is_same_v<decltype(cylinder),
                     std::future<tf::polygons_buffer<tf::cpp::default_index_t,
                                                     TestType, 3, 3>>>);
  static_assert(
      std::is_same_v<decltype(box),
                     std::future<tf::polygons_buffer<tf::cpp::default_index_t,
                                                     TestType, 3, 3>>>);
  static_assert(
      std::is_same_v<decltype(subdivided),
                     std::future<tf::polygons_buffer<tf::cpp::default_index_t,
                                                     TestType, 3, 3>>>);
  static_assert(
      std::is_same_v<decltype(plane),
                     std::future<tf::polygons_buffer<tf::cpp::default_index_t,
                                                     TestType, 3, 3>>>);
  CHECK(submissions == 1);

  check_mesh_shape(sphere.get(), 20, 36);
  check_mesh_shape(cylinder.get(), 18, 32);
  check_mesh_shape(box.get(), 8, 12);
  check_mesh_shape(subdivided.get(), 54, 104);
  const auto actual_plane = plane.get();
  const auto expected_plane =
      tf::cpp::make_plane_mesh(TestType{8}, TestType{6}, 4, 3);
  check_mesh_shape(actual_plane, point_count(expected_plane),
                   face_count(expected_plane));

  auto failure = tf::cpp::async::make_sphere_mesh(TestType{1}, 1, 3);
  CHECK_THROWS_AS(failure.get(), std::invalid_argument);
}

TEMPLATE_TEST_CASE("async tube meshes retain the curve arrays they carry",
                   "[cpp][geometry][mesh-primitives][async][tube][ownership]",
                   float, double) {
  auto source = straight_curve<TestType>();
  const auto expected =
      tf::cpp::make_tube_mesh(source.paths, source.points, TestType{0.25}, 6);
  auto result = tf::cpp::async::make_tube_mesh(source.paths, source.points,
                                               TestType{0.25}, 6);
  static_assert(
      std::is_same_v<decltype(result),
                     std::future<tf::polygons_buffer<tf::cpp::default_index_t,
                                                     TestType, 3, 3>>>);

  source.paths = {};
  source.points.destroy();

  const auto actual = result.get();
  REQUIRE(points_of(actual).size() == points_of(expected).size());
  for (std::size_t index = 0; index < points_of(actual).size(); ++index)
    CHECK(points_of(actual)[index] == points_of(expected)[index]);
}

TEMPLATE_TEST_CASE("mesh spheres preserve dtype, dimensions, and topology",
                   "[cpp][geometry][mesh-primitives]", float, double) {
  auto value = tf::cpp::make_sphere_mesh(TestType{2}, 4, 6);
  static_assert(
      std::is_same_v<decltype(value),
                     primitive_mesh<tf::cpp::default_index_t, TestType>>);
  check_mesh_shape(value, 20, 36);

  const auto &points = points_of(value);
  for (int point = 0; point < point_count(value); ++point) {
    const auto offset = static_cast<std::size_t>(3 * point);
    const auto squared_radius =
        static_cast<double>(points[offset]) * points[offset] +
        static_cast<double>(points[offset + 1]) * points[offset + 1] +
        static_cast<double>(points[offset + 2]) * points[offset + 2];
    CHECK(squared_radius == Catch::Approx(4.0).margin(tolerance<TestType>()));
  }
  CHECK(coordinate_min(value, 2) == TestType{-2});
  CHECK(coordinate_max(value, 2) == TestType{2});
}

TEMPLATE_TEST_CASE("mesh cylinders preserve dtype, dimensions, and topology",
                   "[cpp][geometry][mesh-primitives]", float, double) {
  auto value = tf::cpp::make_cylinder_mesh(TestType{2}, TestType{6}, 8);
  static_assert(
      std::is_same_v<decltype(value),
                     primitive_mesh<tf::cpp::default_index_t, TestType>>);
  check_mesh_shape(value, 18, 32);
  CHECK(coordinate_min(value, 0) ==
        Catch::Approx(-2.0).margin(tolerance<TestType>()));
  CHECK(coordinate_max(value, 0) ==
        Catch::Approx(2.0).margin(tolerance<TestType>()));
  CHECK(coordinate_min(value, 2) == TestType{-3});
  CHECK(coordinate_max(value, 2) == TestType{3});
}

TEMPLATE_TEST_CASE("plain and subdivided boxes preserve canonical dimensions",
                   "[cpp][geometry][mesh-primitives]", float, double) {
  auto plain = tf::cpp::make_box_mesh(TestType{2}, TestType{4}, TestType{6});
  auto subdivided =
      tf::cpp::make_box_mesh(TestType{2}, TestType{4}, TestType{6}, 2, 3, 4);
  static_assert(
      std::is_same_v<decltype(plain),
                     primitive_mesh<tf::cpp::default_index_t, TestType>>);
  static_assert(
      std::is_same_v<decltype(subdivided),
                     primitive_mesh<tf::cpp::default_index_t, TestType>>);
  check_mesh_shape(plain, 8, 12);
  check_mesh_shape(subdivided, 54, 104);

  for (const auto *value : {&plain, &subdivided}) {
    CHECK(coordinate_min(*value, 0) ==
          Catch::Approx(-1.0).margin(tolerance<TestType>()));
    CHECK(coordinate_max(*value, 0) ==
          Catch::Approx(1.0).margin(tolerance<TestType>()));
    CHECK(coordinate_min(*value, 1) ==
          Catch::Approx(-2.0).margin(tolerance<TestType>()));
    CHECK(coordinate_max(*value, 1) ==
          Catch::Approx(2.0).margin(tolerance<TestType>()));
    CHECK(coordinate_min(*value, 2) ==
          Catch::Approx(-3.0).margin(tolerance<TestType>()));
    CHECK(coordinate_max(*value, 2) ==
          Catch::Approx(3.0).margin(tolerance<TestType>()));
  }
}

TEMPLATE_TEST_CASE("mesh planes preserve dtype, dimensions, and topology",
                   "[cpp][geometry][mesh-primitives]", float, double) {
  auto value = tf::cpp::make_plane_mesh(TestType{8}, TestType{6}, 4, 3);
  auto plain = tf::cpp::make_plane_mesh(TestType{8}, TestType{6});
  static_assert(
      std::is_same_v<decltype(value),
                     primitive_mesh<tf::cpp::default_index_t, TestType>>);
  check_mesh_shape(value, 20, 24);
  check_mesh_shape(plain, 4, 2);
  CHECK(coordinate_min(value, 0) == TestType{-4});
  CHECK(coordinate_max(value, 0) == TestType{4});
  CHECK(coordinate_min(value, 1) == TestType{-3});
  CHECK(coordinate_max(value, 1) == TestType{3});
  CHECK(coordinate_min(value, 2) == TestType{0});
  CHECK(coordinate_max(value, 2) == TestType{0});
}

TEMPLATE_TEST_CASE("tube meshes preserve radius, dtype, and open topology",
                   "[cpp][geometry][mesh-primitives][tube]", float, double) {
  auto source = straight_curve<TestType>();
  auto value =
      tf::cpp::make_tube_mesh(source.paths, source.points, TestType{0.25}, 6);
  static_assert(
      std::is_same_v<decltype(value),
                     primitive_mesh<tf::cpp::default_index_t, TestType>>);
  check_mesh_shape(value, 18, 24);

  const auto &points = points_of(value);
  for (int point = 0; point < point_count(value); ++point) {
    const auto offset = static_cast<std::size_t>(3 * point);
    const auto distance =
        std::sqrt(static_cast<double>(points[offset + 1]) * points[offset + 1] +
                  static_cast<double>(points[offset + 2]) * points[offset + 2]);
    CHECK(distance == Catch::Approx(0.25).margin(10 * tolerance<TestType>()));
  }
}

TEMPLATE_TEST_CASE("mesh primitive counts reject unsafe inputs consistently",
                   "[cpp][geometry][mesh-primitives][validation]", float,
                   double) {
  CHECK_THROWS_AS(tf::cpp::make_sphere_mesh(TestType{1}, 1, 3),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::make_sphere_mesh(TestType{1}, -1, 3),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::make_sphere_mesh(TestType{1}, 2, 2),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::make_sphere_mesh(TestType{1}, 2, -1),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::make_cylinder_mesh(TestType{1}, TestType{1}, 2),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::make_cylinder_mesh(TestType{1}, TestType{1}, -1),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::make_box_mesh(TestType{1}, TestType{1}, TestType{1}, 0, 1, 1),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::make_box_mesh(TestType{1}, TestType{1}, TestType{1}, 1, -1, 1),
      std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::make_plane_mesh(TestType{1}, TestType{1}, 1, 0),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::make_plane_mesh(TestType{1}, TestType{1}, -1, 1),
                  std::invalid_argument);

  auto source = straight_curve<TestType>();
  CHECK_THROWS_AS(
      tf::cpp::make_tube_mesh(source.paths, source.points, TestType{1}, -1),
      std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::make_tube_mesh(source.paths, source.points, TestType{1}, 2),
      std::invalid_argument);

  const auto maximum = std::numeric_limits<std::int32_t>::max();
  CHECK_THROWS_AS(tf::cpp::make_sphere_mesh(TestType{1}, maximum, maximum),
                  std::length_error);
  CHECK_THROWS_AS(
      tf::cpp::make_cylinder_mesh(TestType{1}, TestType{1}, maximum),
      std::length_error);
  CHECK_THROWS_AS(tf::cpp::make_box_mesh(TestType{1}, TestType{1}, TestType{1},
                                         maximum, maximum, maximum),
                  std::length_error);
  CHECK_THROWS_AS(
      tf::cpp::make_plane_mesh(TestType{1}, TestType{1}, maximum, maximum),
      std::length_error);
  CHECK_THROWS_AS(tf::cpp::make_tube_mesh(source.paths, source.points,
                                          TestType{1}, maximum),
                  std::length_error);

  check_mesh_shape(tf::cpp::make_sphere_mesh(TestType{1}, 2, 3), 5, 6);
  check_mesh_shape(tf::cpp::make_cylinder_mesh(TestType{1}, TestType{1}, 3), 8,
                   12);
  check_mesh_shape(
      tf::cpp::make_tube_mesh(source.paths, source.points, TestType{1}, 3), 9,
      12);
}

TEMPLATE_TEST_CASE("tube meshes validate handles and preserve empty curves",
                   "[cpp][geometry][mesh-primitives][tube][validation]", float,
                   double) {
  tube_input<TestType> invalid;
  CHECK_THROWS_AS(
      tf::cpp::make_tube_mesh(invalid.paths, invalid.points, TestType{1}, 3),
      std::invalid_argument);

  auto destroyed = straight_curve<TestType>();
  destroyed.paths = {};
  CHECK_THROWS_AS(tf::cpp::make_tube_mesh(destroyed.paths, destroyed.points,
                                          TestType{1}, 3),
                  std::invalid_argument);

  auto empty = empty_curves<TestType>();
  auto singleton = singleton_curve<TestType>();
  check_mesh_shape(
      tf::cpp::make_tube_mesh(empty.paths, empty.points, TestType{1}, 3), 0, 0);
  check_mesh_shape(tf::cpp::make_tube_mesh(singleton.paths, singleton.points,
                                           TestType{1}, 3),
                   0, 0);
}

TEMPLATE_TEST_CASE("tube results own geometry independently of curve inputs",
                   "[cpp][geometry][mesh-primitives][tube][ownership]", float,
                   double) {
  auto source = straight_curve<TestType>();
  auto result =
      tf::cpp::make_tube_mesh(source.paths, source.points, TestType{0.25}, 6);
  const auto first_result_coordinate = points_of(result)[0];

  source.points[0] = TestType{100};
  source.paths = {};
  source.points.destroy();

  CHECK(points_of(result)[0] == first_result_coordinate);
  check_mesh_shape(result, 18, 24);
}

// The deduced spelling and the explicit one are ONE entry now: the index leads
// with the facade's own width as its default, so make_box_mesh(w, h, d) and
// make_box_mesh<std::int32_t>(w, h, d) name the same symbol.
TEMPLATE_TEST_CASE(
    "a mesh primitive is one entry whether or not its index is spelled",
    "[cpp][geometry][mesh-primitives][archive-link][overloads]", float,
    double) {
  using sphere_function =
      auto (*)(TestType, std::int32_t, std::int32_t)
          ->primitive_mesh<tf::cpp::default_index_t, TestType>;
  using box_function = auto (*)(TestType, TestType, TestType)
                           ->primitive_mesh<tf::cpp::default_index_t, TestType>;
  using wide_box_function = auto (*)(TestType, TestType, TestType)
                                ->primitive_mesh<std::int64_t, TestType>;

  const sphere_function deduced_sphere = &tf::cpp::make_sphere_mesh<>;
  const sphere_function typed_sphere =
      &tf::cpp::make_sphere_mesh<std::int32_t, TestType>;
  const box_function deduced_box = &tf::cpp::make_box_mesh<>;
  const box_function typed_box =
      &tf::cpp::make_box_mesh<std::int32_t, TestType>;
  const wide_box_function wide_box =
      &tf::cpp::make_box_mesh<std::int64_t, TestType>;

  CHECK(deduced_sphere == typed_sphere);
  CHECK(deduced_box == typed_box);
  CHECK(reinterpret_cast<const void *>(deduced_box) !=
        reinterpret_cast<const void *>(wide_box));
  CHECK(
      typed_box(TestType{1}, TestType{1}, TestType{1}).faces_buffer().size() ==
      wide_box(TestType{1}, TestType{1}, TestType{1}).faces_buffer().size());
}

TEMPLATE_TEST_CASE("mesh primitive overloads link through native archive",
                   "[cpp][geometry][mesh-primitives][archive-link]", float,
                   double) {
  using sphere_function =
      auto (*)(TestType, std::int32_t, std::int32_t)
          ->primitive_mesh<tf::cpp::default_index_t, TestType>;
  using cylinder_function =
      auto (*)(TestType, TestType, std::int32_t)
          ->primitive_mesh<tf::cpp::default_index_t, TestType>;
  using box_function = auto (*)(TestType, TestType, TestType)
                           ->primitive_mesh<tf::cpp::default_index_t, TestType>;
  using subdivided_box_function =
      auto (*)(TestType, TestType, TestType, std::int32_t, std::int32_t,
               std::int32_t)
          ->primitive_mesh<tf::cpp::default_index_t, TestType>;
  using plane_function =
      auto (*)(TestType, TestType, std::int32_t, std::int32_t)
          ->primitive_mesh<tf::cpp::default_index_t, TestType>;
  using tube_function =
      auto (*)(const tf::cpp::offset_blocked_buffer<tf::cpp::default_index_t,
                                                    tf::cpp::default_index_t> &,
               const tf::cpp::nd_array<TestType> &, TestType, std::int32_t)
          ->primitive_mesh<tf::cpp::default_index_t, TestType>;

  const sphere_function sphere = &tf::cpp::make_sphere_mesh<>;
  const cylinder_function cylinder = &tf::cpp::make_cylinder_mesh<>;
  const box_function box = &tf::cpp::make_box_mesh<>;
  const subdivided_box_function subdivided_box = &tf::cpp::make_box_mesh<>;
  const plane_function plane = &tf::cpp::make_plane_mesh<>;
  const tube_function tube =
      &tf::cpp::make_tube_mesh<tf::cpp::default_index_t, TestType>;
  auto source = straight_curve<TestType>();

  CHECK(sphere(TestType{1}, 2, 3).faces_buffer().size() == 6);
  CHECK(cylinder(TestType{1}, TestType{1}, 3).faces_buffer().size() == 12);
  CHECK(box(TestType{1}, TestType{1}, TestType{1}).faces_buffer().size() == 12);
  CHECK(subdivided_box(TestType{1}, TestType{1}, TestType{1}, 1, 1, 1)
            .faces_buffer()
            .size() == 12);
  CHECK(plane(TestType{1}, TestType{1}, 1, 1).faces_buffer().size() == 2);
  CHECK(
      tube(source.paths, source.points, TestType{1}, 3).faces_buffer().size() ==
      12);
}

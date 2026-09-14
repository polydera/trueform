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
#include "mixed_mesh.hpp"
#include "trueform/cpp/clean.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename Real, std::size_t Dims> struct point_matrix_row {
  using real_type = Real;
  static constexpr std::size_t dims = Dims;
};

using point_matrix_rows =
    std::tuple<point_matrix_row<float, 2>, point_matrix_row<float, 3>,
               point_matrix_row<double, 2>, point_matrix_row<double, 3>>;

template <typename Index, typename Real, std::size_t Dims>
struct form_matrix_row {
  using real_type = Real;
  using index_type = Index;
  using mesh_type = tf::cpp::test::owned_mesh<Index, Real, Dims>;
  using edge_mesh_type = tf::cpp::test::owned_edge_mesh<Index, Real, Dims>;
  using mesh_result_type = tf::polygons_buffer<Index, Real, Dims, 3>;
  using dynamic_mesh_result_type =
      tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size>;
  using edge_result_type = tf::segments_buffer<Index, Real, Dims>;
  static constexpr std::size_t dims = Dims;
};

using form_matrix_rows = std::tuple<form_matrix_row<std::int32_t, float, 2>,
                                    form_matrix_row<std::int32_t, float, 3>,
                                    form_matrix_row<std::int64_t, float, 2>,
                                    form_matrix_row<std::int64_t, float, 3>,
                                    form_matrix_row<std::int32_t, double, 2>,
                                    form_matrix_row<std::int32_t, double, 3>,
                                    form_matrix_row<std::int64_t, double, 2>,
                                    form_matrix_row<std::int64_t, double, 3>>;

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

template <typename Real, std::size_t Dims, typename = void>
struct has_raw_cleaned_points : std::false_type {};
template <typename Real, std::size_t Dims>
struct has_raw_cleaned_points<
    Real, Dims,
    std::void_t<decltype(tf::cpp::cleaned_points<Real, Dims>(
        std::declval<const tf::cpp::nd_array<Real> &>()))>> : std::true_type {};

template <typename Real, std::size_t Dims, typename = void>
struct has_raw_cleaned_points_with_map : std::false_type {};
template <typename Real, std::size_t Dims>
struct has_raw_cleaned_points_with_map<
    Real, Dims,
    std::void_t<decltype(tf::cpp::cleaned_points_with_map<Real, Dims>(
        std::declval<const tf::cpp::nd_array<Real> &>()))>> : std::true_type {};

template <typename Real, std::size_t Dims, typename = void>
struct has_future_raw_cleaned_points : std::false_type {};
template <typename Real, std::size_t Dims>
struct has_future_raw_cleaned_points<
    Real, Dims,
    std::void_t<decltype(tf::cpp::async::cleaned_points<Real, Dims>(
        std::declval<const tf::cpp::nd_array<Real> &>()))>> : std::true_type {};

template <typename Real, std::size_t Dims, typename = void>
struct has_future_raw_cleaned_points_with_map : std::false_type {};
template <typename Real, std::size_t Dims>
struct has_future_raw_cleaned_points_with_map<
    Real, Dims,
    std::void_t<decltype(tf::cpp::async::cleaned_points_with_map<Real, Dims>(
        std::declval<const tf::cpp::nd_array<Real> &>()))>> : std::true_type {};

template <typename Real, std::size_t Dims, typename = void>
struct has_resolver_raw_cleaned_points : std::false_type {};
template <typename Real, std::size_t Dims>
struct has_resolver_raw_cleaned_points<
    Real, Dims,
    std::void_t<decltype(tf::cpp::async::cleaned_points<Real, Dims>(
        std::declval<mutating_resolver>(),
        std::declval<const tf::cpp::nd_array<Real> &>()))>> : std::true_type {};

template <typename Real, std::size_t Dims, typename = void>
struct has_resolver_raw_cleaned_points_with_map : std::false_type {};
template <typename Real, std::size_t Dims>
struct has_resolver_raw_cleaned_points_with_map<
    Real, Dims,
    std::void_t<decltype(tf::cpp::async::cleaned_points_with_map<Real, Dims>(
        std::declval<mutating_resolver>(),
        std::declval<const tf::cpp::nd_array<Real> &>()))>> : std::true_type {};

template <typename Real, std::size_t Dims>
inline constexpr bool has_all_raw_cleaned_points_v =
    has_raw_cleaned_points<Real, Dims>::value &&
    has_raw_cleaned_points_with_map<Real, Dims>::value &&
    has_future_raw_cleaned_points<Real, Dims>::value &&
    has_future_raw_cleaned_points_with_map<Real, Dims>::value &&
    has_resolver_raw_cleaned_points<Real, Dims>::value &&
    has_resolver_raw_cleaned_points_with_map<Real, Dims>::value;

template <typename Real, std::size_t Dims>
inline constexpr bool has_no_raw_cleaned_points_v =
    !has_raw_cleaned_points<Real, Dims>::value &&
    !has_raw_cleaned_points_with_map<Real, Dims>::value &&
    !has_future_raw_cleaned_points<Real, Dims>::value &&
    !has_future_raw_cleaned_points_with_map<Real, Dims>::value &&
    !has_resolver_raw_cleaned_points<Real, Dims>::value &&
    !has_resolver_raw_cleaned_points_with_map<Real, Dims>::value;

template <typename Real, std::size_t Dims, typename = void>
struct has_resolver_point_cloud_cleaned_points : std::false_type {};
template <typename Real, std::size_t Dims>
struct has_resolver_point_cloud_cleaned_points<
    Real, Dims,
    std::void_t<decltype(tf::cpp::async::cleaned_points(
        std::declval<mutating_resolver>(),
        std::declval<const tf::cpp::point_cloud<Real, Dims> &>()))>>
    : std::true_type {};

template <typename Real, std::size_t Dims, typename = void>
struct has_resolver_point_cloud_cleaned_points_with_map : std::false_type {};
template <typename Real, std::size_t Dims>
struct has_resolver_point_cloud_cleaned_points_with_map<
    Real, Dims,
    std::void_t<decltype(tf::cpp::async::cleaned_points_with_map(
        std::declval<mutating_resolver>(),
        std::declval<const tf::cpp::point_cloud<Real, Dims> &>()))>>
    : std::true_type {};

template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices,
          typename = void>
struct has_cleaned_polygon_soup_result : std::false_type {};
template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices>
struct has_cleaned_polygon_soup_result<
    Index, Real, Dims, Vertices,
    std::void_t<tf::cpp::basic_cleaned_polygon_soup_result<Index, Real, Dims,
                                                           Vertices>>>
    : std::true_type {};

template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices,
          typename = void>
struct has_typed_cleaned_polygon_soup : std::false_type {};
template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices>
struct has_typed_cleaned_polygon_soup<
    Index, Real, Dims, Vertices,
    std::void_t<
        decltype(tf::cpp::cleaned_polygon_soup<Index, Real, Dims, Vertices>(
            std::declval<const tf::cpp::nd_array<Real> &>()))>>
    : std::true_type {};

template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices,
          typename = void>
struct has_future_typed_cleaned_polygon_soup : std::false_type {};
template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices>
struct has_future_typed_cleaned_polygon_soup<
    Index, Real, Dims, Vertices,
    std::void_t<decltype(tf::cpp::async::cleaned_polygon_soup<Index, Real, Dims,
                                                              Vertices>(
        std::declval<const tf::cpp::nd_array<Real> &>()))>> : std::true_type {};

template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices,
          typename = void>
struct has_resolver_typed_cleaned_polygon_soup : std::false_type {};
template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices>
struct has_resolver_typed_cleaned_polygon_soup<
    Index, Real, Dims, Vertices,
    std::void_t<decltype(tf::cpp::async::cleaned_polygon_soup<Index, Real, Dims,
                                                              Vertices>(
        std::declval<mutating_resolver>(),
        std::declval<const tf::cpp::nd_array<Real> &>()))>> : std::true_type {};

template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices>
inline constexpr bool has_all_typed_cleaned_polygon_soup_v =
    has_cleaned_polygon_soup_result<Index, Real, Dims, Vertices>::value &&
    has_typed_cleaned_polygon_soup<Index, Real, Dims, Vertices>::value &&
    has_future_typed_cleaned_polygon_soup<Index, Real, Dims, Vertices>::value &&
    has_resolver_typed_cleaned_polygon_soup<Index, Real, Dims, Vertices>::value;

template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices>
inline constexpr bool has_no_typed_cleaned_polygon_soup_v =
    !has_cleaned_polygon_soup_result<Index, Real, Dims, Vertices>::value &&
    !has_typed_cleaned_polygon_soup<Index, Real, Dims, Vertices>::value &&
    !has_future_typed_cleaned_polygon_soup<Index, Real, Dims,
                                           Vertices>::value &&
    !has_resolver_typed_cleaned_polygon_soup<Index, Real, Dims,
                                             Vertices>::value;

static_assert(tf::cpp::is_supported_cleaned_points_v<float, 2>);
static_assert(tf::cpp::is_supported_cleaned_points_v<double, 3>);
static_assert(!tf::cpp::is_supported_cleaned_points_v<long double, 2>);
static_assert(!tf::cpp::is_supported_cleaned_points_v<const float, 2>);
static_assert(!tf::cpp::is_supported_cleaned_points_v<float, 4>);
static_assert(has_all_raw_cleaned_points_v<float, 2>);
static_assert(has_all_raw_cleaned_points_v<double, 3>);
static_assert(has_no_raw_cleaned_points_v<long double, 2>);
static_assert(has_no_raw_cleaned_points_v<const float, 3>);
static_assert(has_no_raw_cleaned_points_v<volatile double, 2>);
static_assert(has_no_raw_cleaned_points_v<float, 4>);
static_assert(has_resolver_point_cloud_cleaned_points<float, 2>::value);
static_assert(
    has_resolver_point_cloud_cleaned_points_with_map<double, 3>::value);
static_assert(!has_resolver_point_cloud_cleaned_points<long double, 2>::value);
static_assert(
    !has_resolver_point_cloud_cleaned_points_with_map<long double, 2>::value);

static_assert(
    tf::cpp::is_supported_cleaned_polygon_soup_v<std::int32_t, float, 2, 2>);
static_assert(
    tf::cpp::is_supported_cleaned_polygon_soup_v<std::int64_t, double, 3, 3>);
static_assert(has_all_typed_cleaned_polygon_soup_v<std::int32_t, float, 2, 2>);
static_assert(has_all_typed_cleaned_polygon_soup_v<std::int64_t, double, 3, 3>);
static_assert(
    has_no_typed_cleaned_polygon_soup_v<std::int32_t, long double, 2, 2>);
static_assert(
    has_no_typed_cleaned_polygon_soup_v<std::int32_t, const float, 3, 3>);
static_assert(
    has_no_typed_cleaned_polygon_soup_v<std::int64_t, volatile double, 2, 3>);
static_assert(has_no_typed_cleaned_polygon_soup_v<std::uint32_t, float, 2, 2>);
static_assert(
    has_no_typed_cleaned_polygon_soup_v<const std::int64_t, double, 3, 3>);
static_assert(has_no_typed_cleaned_polygon_soup_v<std::int32_t, float, 4, 2>);
static_assert(has_no_typed_cleaned_polygon_soup_v<std::int64_t, double, 3, 4>);

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

template <typename T>
auto same_array(const tf::cpp::nd_array<T> &first,
                const tf::cpp::nd_array<T> &second) -> bool {
  if (first.raw_shape() != second.raw_shape() ||
      first.length() != second.length())
    return false;
  for (std::size_t index = 0; index < first.length(); ++index)
    if (first[index] != second[index])
      return false;
  return true;
}

/// A result is core's own storage, so a check reads the flat arrays it lies in.
template <typename Real, std::size_t Dims>
auto points_array(const tf::points_buffer<Real, Dims> &value)
    -> tf::cpp::nd_array<Real> {
  return tf::cpp::test::copied_nd_array(
      value.data_buffer(),
      {static_cast<int>(value.size()), static_cast<int>(Dims)});
}

template <typename Index, typename Real, std::size_t Dims>
auto edges_array(const tf::segments_buffer<Index, Real, Dims> &value)
    -> tf::cpp::nd_array<Index> {
  return tf::cpp::test::copied_nd_array(value.edges_buffer().data_buffer(),
                                        {static_cast<int>(value.size()), 2});
}

template <typename Real> auto duplicate_points() -> tf::cpp::nd_array<Real> {
  return make_array<Real>(
      {0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 1, 0, 1, 1, Real{0.5}, 2, 1, 0}, {6, 3});
}

template <typename Real, std::size_t Dims>
auto python_duplicate_points() -> tf::cpp::nd_array<Real> {
  if constexpr (Dims == 2)
    return make_array<Real>({0, 0, 1, 0, 0, 0, 2, 1, 1, 1, 2, 1}, {6, 2});
  else
    return duplicate_points<Real>();
}

template <typename Real, std::size_t Dims>
auto python_tolerance_points() -> tf::cpp::nd_array<Real> {
  if constexpr (Dims == 2)
    return make_array<Real>(
        {0, 0, Real{0.001}, Real{0.001}, 1, 0, Real{1.001}, 0, 2, 2}, {5, 2});
  else
    return make_array<Real>(
        {0, 0, 0, Real{0.001}, 0, 0, 1, 0, 0, 1, Real{0.001}, 0, 2, 2, 2},
        {5, 3});
}

template <typename Real, std::size_t Dims>
auto python_noop_points() -> tf::cpp::nd_array<Real> {
  if constexpr (Dims == 2)
    return make_array<Real>({0, 0, 1, 0, 0, 1}, {3, 2});
  else
    return make_array<Real>({0, 0, 0, 1, 0, 0, 0, 1, 0}, {3, 3});
}

template <typename Real>
auto duplicate_mesh()
    -> tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 3, 1, 4}, {0, 0, 0, 1, 0, 0, Real{0.5}, 1, 0, 0, 0, 0,
                           Real{1.5}, Real{0.5}, Real{0.5}})};
}

template <typename Real>
auto near_duplicate_points() -> tf::cpp::nd_array<Real> {
  return make_array<Real>({0, 0, 0, Real{0.009}, 0, 0, 1, 1, 0}, {3, 3});
}

template <typename Real>
auto near_degenerate_mesh()
    -> tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2}, {0, 0, 0, Real{0.009}, 0, 0, 1, 1, 0})};
}

template <typename Real>
auto near_degenerate_soup() -> tf::cpp::nd_array<Real> {
  auto points = near_duplicate_points<Real>();
  points.set_shape({1, 3, 3});
  return points;
}

template <typename Row>
auto form_points_with_duplicate() -> std::vector<typename Row::real_type> {
  using Real = typename Row::real_type;
  if constexpr (Row::dims == 2)
    return {0, 0, 1, 0, Real{0.5}, 1, 0, 0, Real{1.5}, Real{0.5}, 9, 9};
  else
    return {0, 0, 0, 1,         0,         0,         Real{0.5}, 1, 0,
            0, 0, 0, Real{1.5}, Real{0.5}, Real{0.5}, 9,         9, 9};
}

template <typename Row>
auto python_edge_mesh() -> typename Row::edge_mesh_type {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  const auto points = [&]() -> std::vector<Real> {
    if constexpr (Row::dims == 2)
      return {0, 0, 1, 0, 0, 0, 1, 1};
    else
      return {0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0};
  }();
  return {tf::cpp::test::segments_of<Index, Real, Row::dims>(
      std::vector<Index>{0, 1, 2, 3}, points)};
}

template <typename Row> auto python_fixed_mesh() -> typename Row::mesh_type {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  const auto points = [&]() -> std::vector<Real> {
    if constexpr (Row::dims == 2)
      return {0, 0, 1, 0, Real{0.5}, 1, 0, 0, Real{1.5}, Real{0.5}};
    else
      return {0, 0, 0, 1, 0,         0,         Real{0.5}, 1,
              0, 0, 0, 0, Real{1.5}, Real{0.5}, Real{0.5}};
  }();
  return {tf::cpp::test::polygons_of<Index, Real, Row::dims>(
      std::vector<Index>{0, 1, 2, 3, 1, 4}, points)};
}

template <typename Row>
auto python_dynamic_mesh()
    -> tf::cpp::test::mixed_mesh_of<typename Row::mesh_type> {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  return {tf::cpp::test::polygons_of<Index, Real, Row::dims>(
      std::vector<Index>{0, 3, 7}, std::vector<Index>{0, 1, 2, 3, 1, 4, 5},
      form_points_with_duplicate<Row>())};
}

template <typename Row, std::size_t Vertices>
auto python_polygon_soup(bool near_shared = false)
    -> tf::cpp::nd_array<typename Row::real_type> {
  using Real = typename Row::real_type;
  static_assert(Vertices == 2 || Vertices == 3);
  const auto shared = near_shared ? Real{1.001} : Real{1};
  if constexpr (Vertices == 2 && Row::dims == 2)
    return make_array<Real>({0, 0, 1, 0, shared, 0, 1, 1}, {2, 2, 2});
  else if constexpr (Vertices == 2 && Row::dims == 3)
    return make_array<Real>({0, 0, 0, 1, 0, 0, shared, 0, 0, 1, 1, 0},
                            {2, 2, 3});
  else if constexpr (Vertices == 3 && Row::dims == 2)
    return make_array<Real>(
        {0, 0, 1, 0, Real{0.5}, 1, shared, 0, 2, 0, Real{1.5}, 1}, {2, 3, 2});
  else
    return make_array<Real>({0, 0, 0, 1, 0, 0, Real{0.5}, 1, 0, shared, 0, 0, 2,
                             0, 0, Real{1.5}, 1, 0},
                            {2, 3, 3});
}

template <std::size_t Vertices, typename Result>
auto soup_connectivity(const Result &result) {
  if constexpr (Vertices == 2)
    return edges_array(result);
  else
    return tf::cpp::test::copied_nd_array(result.faces_buffer().data_buffer(),
                                          {static_cast<int>(result.size()), 3});
}

template <typename Result> auto soup_points(const Result &result) {
  return points_array(result.points_buffer());
}

template <std::size_t Vertices, typename Result>
auto soup_primitive_count(const Result &result) -> int {
  static_cast<void>(Vertices);
  return static_cast<int>(result.size());
}

template <typename Result> auto soup_point_count(const Result &result) -> int {
  return static_cast<int>(result.points_buffer().size());
}

template <std::size_t Vertices, std::size_t Dims, typename Index, typename Real>
auto soup_has_same_oriented_primitives(
    const tf::cpp::nd_array<Real> &input,
    const tf::cpp::nd_array<Index> &connectivity,
    const tf::cpp::nd_array<Real> &points, Real tolerance) -> bool {
  const auto primitive_count = input.ndim() == 2 ? 1 : input.shape_at(0);
  if (connectivity.ndim() != 2 || connectivity.shape_at(0) != primitive_count ||
      connectivity.shape_at(1) != static_cast<int>(Vertices) ||
      points.ndim() != 2 || points.shape_at(1) != static_cast<int>(Dims))
    return false;

  std::vector<bool> matched(static_cast<std::size_t>(primitive_count), false);
  const auto tolerance2 = tolerance * tolerance;
  for (int source = 0; source < primitive_count; ++source) {
    auto found = false;
    for (int output = 0; output < primitive_count && !found; ++output) {
      if (matched[static_cast<std::size_t>(output)])
        continue;
      for (std::size_t shift = 0; shift < Vertices && !found; ++shift) {
        auto same = true;
        for (std::size_t corner = 0; corner < Vertices && same; ++corner) {
          const auto output_corner = (corner + shift) % Vertices;
          const auto point_id =
              connectivity[static_cast<std::size_t>(output) * Vertices +
                           output_corner];
          if (point_id < Index{0} || point_id >= points.shape_at(0)) {
            same = false;
            break;
          }
          auto distance2 = Real{0};
          for (std::size_t coordinate = 0; coordinate < Dims; ++coordinate) {
            const auto delta =
                input[(static_cast<std::size_t>(source) * Vertices + corner) *
                          Dims +
                      coordinate] -
                points[static_cast<std::size_t>(point_id) * Dims + coordinate];
            distance2 += delta * delta;
          }
          same = tolerance == Real{0} ? distance2 == Real{0}
                                      : distance2 <= tolerance2;
        }
        if (same) {
          matched[static_cast<std::size_t>(output)] = true;
          found = true;
        }
      }
    }
    if (!found)
      return false;
  }
  return true;
}

template <typename Row>
auto local_transform()
    -> std::array<typename Row::real_type, (Row::dims + 1) * (Row::dims + 1)> {
  using Real = typename Row::real_type;
  if constexpr (Row::dims == 2)
    return {100, 0, 7, 0, 100, 8, 0, 0, 1};
  else
    return {100, 0, 0, 7, 0, 100, 0, 8, 0, 0, 100, 9, 0, 0, 0, Real{1}};
}

template <typename Row>
auto empty_dynamic_mesh()
    -> tf::cpp::test::mixed_mesh_of<typename Row::mesh_type> {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  const auto points = [&]() -> std::vector<Real> {
    if constexpr (Row::dims == 2)
      return {0, 0, 0, 0, 1, 0};
    else
      return {0, 0, 0, 0, 0, 0, 1, 0, 0};
  }();
  return {tf::cpp::test::polygons_of<Index, Real, Row::dims>(
      std::vector<Index>{0}, std::vector<Index>{}, points)};
}

template <typename Row>
auto cleanup_edge_mesh() -> typename Row::edge_mesh_type {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  return {tf::cpp::test::segments_of<Index, Real, Row::dims>(
      std::vector<Index>{0, 1, 3, 1, 0, 3, 1, 4},
      form_points_with_duplicate<Row>())};
}

template <typename Row> auto cleanup_fixed_mesh() -> typename Row::mesh_type {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  return {tf::cpp::test::polygons_of<Index, Real, Row::dims>(
      std::vector<Index>{0, 1, 4, 3, 1, 4, 0, 1, 3},
      form_points_with_duplicate<Row>())};
}

template <typename Row, std::size_t Vertices>
auto first_soup_primitive() -> tf::cpp::nd_array<typename Row::real_type> {
  using Real = typename Row::real_type;
  const auto batch = python_polygon_soup<Row, Vertices>();
  tf::buffer<Real> data;
  data.allocate(Vertices * Row::dims);
  std::copy_n(batch.begin(), Vertices * Row::dims, data.begin());
  return tf::cpp::nd_array<Real>::from_buffer(
      std::move(data),
      {static_cast<int>(Vertices), static_cast<int>(Row::dims)});
}

template <typename Row, std::size_t Vertices>
auto check_python_polygon_soup_fixture() -> void {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  using Result =
      tf::cpp::basic_cleaned_polygon_soup_result<Index, Real, Row::dims,
                                                 Vertices>;

  auto input = python_polygon_soup<Row, Vertices>();
  const auto original = input.deep_copy();
  const auto exact =
      tf::cpp::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(input);
  const auto direct =
      tf::cpp::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(
          input, Real{}, false, false);
  const auto connectivity = soup_connectivity<Vertices>(exact);

  STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(exact)>, Result>);
  STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(connectivity)>,
                                tf::cpp::nd_array<Index>>);
  REQUIRE(soup_primitive_count<Vertices>(exact) == 2);
  REQUIRE(soup_point_count(exact) == (Vertices == 2 ? 3 : 5));
  REQUIRE((connectivity.raw_shape() ==
           tf::small_vector<int, 3>{2, static_cast<int>(Vertices)}));
  const auto exact_points = soup_points(exact);
  REQUIRE((exact_points.raw_shape() ==
           tf::small_vector<int, 3>{soup_point_count(exact),
                                    static_cast<int>(Row::dims)}));
  CHECK(soup_has_same_oriented_primitives<Vertices, Row::dims>(
      original, connectivity, exact_points, Real{}));

  std::vector<bool> referenced(
      static_cast<std::size_t>(soup_point_count(exact)), false);
  for (const auto point_id : connectivity) {
    REQUIRE(point_id >= Index{0});
    REQUIRE(point_id < soup_point_count(exact));
    referenced[static_cast<std::size_t>(point_id)] = true;
  }
  CHECK(std::all_of(referenced.begin(), referenced.end(),
                    [](const auto value) { return value; }));
  CHECK(same_array(input, original));
  CHECK(same_array(soup_connectivity<Vertices>(direct), connectivity));
  CHECK(same_array(soup_points(direct), exact_points));

  input[0] = Real{99};
  CHECK(same_array(soup_points(exact), exact_points));

  auto near = python_polygon_soup<Row, Vertices>(true);
  const auto exact_near =
      tf::cpp::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(near);
  const auto tolerance =
      tf::cpp::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(
          near, Real{0.01});
  REQUIRE(soup_point_count(exact_near) == static_cast<int>(2 * Vertices));
  REQUIRE(soup_point_count(tolerance) == (Vertices == 2 ? 3 : 5));
  REQUIRE(soup_primitive_count<Vertices>(tolerance) == 2);
  CHECK(soup_has_same_oriented_primitives<Vertices, Row::dims>(
      near, soup_connectivity<Vertices>(tolerance), soup_points(tolerance),
      Real{0.01}));

  const auto repeated =
      tf::cpp::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(original);
  CHECK(same_array(soup_connectivity<Vertices>(repeated), connectivity));
  CHECK(same_array(soup_points(repeated), exact_points));
}

template <typename Row, std::size_t Vertices>
auto check_polygon_soup_boundaries() -> void {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  using Result =
      tf::cpp::basic_cleaned_polygon_soup_result<Index, Real, Row::dims,
                                                 Vertices>;

  auto empty = make_array<Real>(
      {}, {0, static_cast<int>(Vertices), static_cast<int>(Row::dims)});
  const auto empty_result =
      tf::cpp::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(empty);
  CHECK(soup_primitive_count<Vertices>(empty_result) == 0);
  CHECK(soup_point_count(empty_result) == 0);
  CHECK((soup_connectivity<Vertices>(empty_result).raw_shape() ==
         tf::small_vector<int, 3>{0, static_cast<int>(Vertices)}));
  CHECK((soup_points(empty_result).raw_shape() ==
         tf::small_vector<int, 3>{0, static_cast<int>(Row::dims)}));

  auto single = first_soup_primitive<Row, Vertices>();
  const auto single_result =
      tf::cpp::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(single);
  CHECK(soup_primitive_count<Vertices>(single_result) == 1);
  CHECK(soup_point_count(single_result) == static_cast<int>(Vertices));
  CHECK(soup_has_same_oriented_primitives<Vertices, Row::dims>(
      single, soup_connectivity<Vertices>(single_result),
      soup_points(single_result), Real{}));

  const tf::cpp::nd_array<Real> invalid;
  const auto wrong_rank = make_array<Real>({Real{0}}, {1});
  CHECK_THROWS_AS(
      (tf::cpp::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(
          invalid)),
      std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(
          wrong_rank)),
      std::invalid_argument);
  CHECK_THROWS_MATCHES(
      (tf::cpp::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(
          single, Real{-1})),
      std::invalid_argument,
      Catch::Matchers::Message("clean: tolerance must be non-negative"));
  CHECK_THROWS_AS(
      (tf::cpp::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(
          single, std::numeric_limits<Real>::infinity())),
      std::invalid_argument);

  auto async_input = python_polygon_soup<Row, Vertices>();
  const auto expected =
      tf::cpp::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(
          async_input);
  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto pending =
      tf::cpp::async::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(
          mutating_resolver{submissions, [&] { async_input.destroy(); }},
          async_input);
  STATIC_REQUIRE(std::is_same_v<decltype(pending), std::future<Result>>);
  const auto async_result = pending.get();
  CHECK(same_array(soup_connectivity<Vertices>(async_result),
                   soup_connectivity<Vertices>(expected)));
  CHECK(same_array(soup_points(async_result), soup_points(expected)));
  CHECK(submissions->load(std::memory_order_relaxed) == 1);

  auto destroyed_input = python_polygon_soup<Row, Vertices>();
  auto destroyed_pending =
      tf::cpp::async::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(
          destroyed_input);
  destroyed_input.destroy();
  CHECK(soup_primitive_count<Vertices>(destroyed_pending.get()) == 2);

  auto invalid_pending =
      tf::cpp::async::cleaned_polygon_soup<Index, Real, Row::dims, Vertices>(
          invalid);
  CHECK_THROWS_AS(invalid_pending.get(), std::invalid_argument);
}

} // namespace

TEMPLATE_LIST_TEST_CASE("Python-parity typed point arrays reproduce exact "
                        "duplicate and tolerance fixtures",
                        "[cpp][clean][python-parity][points][matrix]",
                        point_matrix_rows) {
  using Real = typename TestType::real_type;
  constexpr auto Dims = TestType::dims;

  auto input = python_duplicate_points<Real, Dims>();
  const auto original = input.deep_copy();
  const auto exact = tf::cpp::cleaned_points_with_map<Real, Dims>(input);
  const auto direct =
      tf::cpp::cleaned_points<Real, Dims>(input, Real{}, false, false);

  REQUIRE((exact.points.raw_shape() ==
           tf::small_vector<int, 3>{4, static_cast<int>(Dims)}));
  REQUIRE(exact.point_map.f.length() == 6);
  REQUIRE(exact.point_map.kept_ids.length() == 4);
  CHECK(exact.point_map.f[0] == exact.point_map.f[2]);
  CHECK(exact.point_map.f[3] == exact.point_map.f[5]);
  CHECK(exact.point_map.f[0] != exact.point_map.f[1]);
  CHECK(exact.point_map.f[1] != exact.point_map.f[4]);
  CHECK(same_array(direct, exact.points));

  for (std::size_t output_id = 0; output_id < exact.point_map.kept_ids.length();
       ++output_id) {
    const auto source_id = exact.point_map.kept_ids[output_id];
    REQUIRE(source_id >= 0);
    REQUIRE(source_id < input.shape_at(0));
    CHECK(exact.point_map.f[static_cast<std::size_t>(source_id)] ==
          static_cast<std::int32_t>(output_id));
    for (std::size_t coordinate = 0; coordinate < Dims; ++coordinate)
      CHECK(exact.points[output_id * Dims + coordinate] ==
            input[static_cast<std::size_t>(source_id) * Dims + coordinate]);
  }

  CHECK(same_array(input, original));

  auto tolerance_input = python_tolerance_points<Real, Dims>();
  const auto tolerance_original = tolerance_input.deep_copy();
  const auto tolerance =
      tf::cpp::cleaned_points_with_map<Real, Dims>(tolerance_input, Real{0.01});
  REQUIRE(tolerance.points.shape_at(0) == 3);
  CHECK(tolerance.point_map.f[0] == tolerance.point_map.f[1]);
  CHECK(tolerance.point_map.f[2] == tolerance.point_map.f[3]);
  CHECK(tolerance.point_map.f[0] != tolerance.point_map.f[2]);
  CHECK(tolerance.point_map.f[2] != tolerance.point_map.f[4]);
  CHECK(same_array(tolerance_input, tolerance_original));

  const auto repeated = tf::cpp::cleaned_points_with_map<Real, Dims>(input);
  CHECK(same_array(repeated.points, exact.points));
  CHECK(same_array(repeated.point_map.f, exact.point_map.f));
  CHECK(same_array(repeated.point_map.kept_ids, exact.point_map.kept_ids));

  if constexpr (Dims == 3) {
    const auto legacy = tf::cpp::cleaned_points_with_map(input);
    CHECK(same_array(legacy.points, exact.points));
    CHECK(same_array(legacy.point_map.f, exact.point_map.f));
    CHECK(same_array(legacy.point_map.kept_ids, exact.point_map.kept_ids));
  }
}

TEMPLATE_LIST_TEST_CASE(
    "Python-parity typed point cleaning handles no-op empty singleton and "
    "validation",
    "[cpp][clean][python-parity][points][matrix][validation]",
    point_matrix_rows) {
  using Real = typename TestType::real_type;
  constexpr auto Dims = TestType::dims;

  auto noop = python_noop_points<Real, Dims>();
  const auto noop_original = noop.deep_copy();
  const auto noop_result = tf::cpp::cleaned_points_with_map<Real, Dims>(noop);
  REQUIRE(noop_result.points.shape_at(0) == 3);
  REQUIRE(noop_result.point_map.kept_ids.length() == 3);
  for (std::size_t output_id = 0; output_id < 3; ++output_id) {
    const auto source_id = noop_result.point_map.kept_ids[output_id];
    REQUIRE(source_id >= 0);
    REQUIRE(source_id < 3);
    for (std::size_t coordinate = 0; coordinate < Dims; ++coordinate)
      CHECK(noop_result.points[output_id * Dims + coordinate] ==
            noop[static_cast<std::size_t>(source_id) * Dims + coordinate]);
  }
  CHECK(same_array(noop, noop_original));

  auto empty = make_array<Real>({}, {0, static_cast<int>(Dims)});
  const auto empty_result = tf::cpp::cleaned_points_with_map<Real, Dims>(empty);
  CHECK((empty_result.points.raw_shape() ==
         tf::small_vector<int, 3>{0, static_cast<int>(Dims)}));
  CHECK(empty_result.point_map.f.length() == 0);
  CHECK(empty_result.point_map.kept_ids.length() == 0);

  auto singleton = make_array<Real>(
      Dims == 2 ? std::initializer_list<Real>{Real{2}, Real{3}}
                : std::initializer_list<Real>{Real{2}, Real{3}, Real{4}},
      {static_cast<int>(Dims)});
  const auto singleton_result =
      tf::cpp::cleaned_points_with_map<Real, Dims>(singleton);
  CHECK((singleton_result.points.raw_shape() ==
         tf::small_vector<int, 3>{1, static_cast<int>(Dims)}));
  CHECK(singleton_result.point_map.f.length() == 1);
  CHECK(singleton_result.point_map.kept_ids.length() == 1);
  CHECK(singleton_result.point_map.f[0] == 0);
  CHECK(singleton_result.point_map.kept_ids[0] == 0);

  CHECK_THROWS_MATCHES(
      (tf::cpp::cleaned_points<Real, Dims>(noop, Real{-1})),
      std::invalid_argument,
      Catch::Matchers::Message("clean: tolerance must be non-negative"));
}

TEST_CASE(
    "Python-parity typed point cleaning rejects invalid ranks and dimensions",
    "[cpp][clean][python-parity][points][matrix][validation]") {
  const auto invalid = tf::cpp::nd_array<float>{};
  const auto rank_three = make_array<float>({0, 1}, {1, 1, 2});
  const auto wrong_2d_batch = make_array<float>({0, 1, 2}, {1, 3});
  const auto wrong_2d_single = make_array<float>({0, 1, 2}, {3});
  const auto wrong_3d_batch = make_array<float>({0, 1}, {1, 2});

  CHECK_THROWS_AS((tf::cpp::cleaned_points<float, 2>(invalid)),
                  std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::cleaned_points<float, 2>(rank_three)),
                  std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::cleaned_points<float, 2>(wrong_2d_batch)),
                  std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::cleaned_points_with_map<float, 2>(wrong_2d_single)),
                  std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::cleaned_points<float, 3>(wrong_3d_batch)),
                  std::invalid_argument);
}

TEMPLATE_LIST_TEST_CASE(
    "Python-parity PointCloud cleaning uses raw points and returns owning maps",
    "[cpp][clean][python-parity][point-cloud][matrix]", point_matrix_rows) {
  using Real = typename TestType::real_type;
  constexpr auto Dims = TestType::dims;

  auto input = python_duplicate_points<Real, Dims>();
  const auto original = input.deep_copy();
  tf::cpp::test::owned_point_cloud<Real, Dims> cloud{
      tf::cpp::test::points_of<Real, Dims>(input.make_range())};
  const auto result = tf::cpp::cleaned_points_with_map(cloud.point_cloud());
  const auto direct = tf::cpp::cleaned_points(cloud.point_cloud());

  CHECK((result.points.raw_shape() ==
         tf::small_vector<int, 3>{4, static_cast<int>(Dims)}));
  CHECK(result.point_map.f.length() == 6);
  CHECK(result.point_map.kept_ids.length() == 4);
  CHECK(result.point_map.f[0] == result.point_map.f[2]);
  CHECK(result.point_map.f[3] == result.point_map.f[5]);
  CHECK(same_array(direct, result.points));
  CHECK(same_array(input, original));

  cloud.points.data_buffer()[0] = Real{99};
  cloud.cache.points_changed();
  CHECK(result.points[0] != Real{99});

  tf::cpp::test::owned_point_cloud<Real, Dims> tolerance_cloud{
      tf::cpp::test::points_of<Real, Dims>(
          python_tolerance_points<Real, Dims>().make_range())};
  CHECK(tf::cpp::cleaned_points(tolerance_cloud.point_cloud(), Real{0.01})
            .shape_at(0) == 3);
  CHECK_THROWS_MATCHES(
      tf::cpp::cleaned_points(tolerance_cloud.point_cloud(), Real{-1}),
      std::invalid_argument,
      Catch::Matchers::Message("clean: tolerance must be non-negative"));
}

/// An array is copied at the call, so a mutation between submission and run is
/// not what the job reads; a cloud is one reading of the caller's own memory,
/// which the caller keeps alive until the future completes.
TEMPLATE_LIST_TEST_CASE(
    "Point cleaning async retains its arrays and reads the caller's cloud",
    "[cpp][clean][python-parity][points][point-cloud][async][ownership]",
    point_matrix_rows) {
  using Real = typename TestType::real_type;
  constexpr auto Dims = TestType::dims;
  using result_type =
      tf::cpp::cleaned_points_result<tf::cpp::default_index_t, Real>;

  auto array = python_duplicate_points<Real, Dims>();
  const auto expected_array =
      tf::cpp::cleaned_points_with_map<Real, Dims>(array);
  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto array_pending = tf::cpp::async::cleaned_points_with_map<Real, Dims>(
      mutating_resolver{submissions, [&] { array.destroy(); }}, array);
  STATIC_REQUIRE(
      std::is_same_v<decltype(array_pending), std::future<result_type>>);
  const auto array_result = array_pending.get();
  CHECK(same_array(array_result.points, expected_array.points));
  CHECK(same_array(array_result.point_map.f, expected_array.point_map.f));
  CHECK(same_array(array_result.point_map.kept_ids,
                   expected_array.point_map.kept_ids));

  tf::cpp::test::owned_point_cloud<Real, Dims> cloud{
      tf::cpp::test::points_of<Real, Dims>(
          python_duplicate_points<Real, Dims>().make_range())};
  const auto expected_cloud =
      tf::cpp::cleaned_points_with_map(cloud.point_cloud());
  auto cloud_pending = tf::cpp::async::cleaned_points_with_map(
      mutating_resolver{submissions, [] {}}, cloud.point_cloud());
  STATIC_REQUIRE(
      std::is_same_v<decltype(cloud_pending), std::future<result_type>>);
  const auto cloud_result = cloud_pending.get();
  CHECK(same_array(cloud_result.points, expected_cloud.points));
  CHECK(same_array(cloud_result.point_map.f, expected_cloud.point_map.f));
  CHECK(same_array(cloud_result.point_map.kept_ids,
                   expected_cloud.point_map.kept_ids));

  if constexpr (Dims == 3) {
    auto legacy = duplicate_points<Real>();
    const auto expected_legacy = tf::cpp::cleaned_points(legacy);
    auto legacy_pending = tf::cpp::async::cleaned_points(
        mutating_resolver{submissions, [&] { legacy.destroy(); }}, legacy);
    STATIC_REQUIRE(std::is_same_v<decltype(legacy_pending),
                                  std::future<tf::cpp::nd_array<Real>>>);
    CHECK(same_array(legacy_pending.get(), expected_legacy));
    CHECK(submissions->load(std::memory_order_relaxed) == 3);
  } else {
    CHECK(submissions->load(std::memory_order_relaxed) == 2);
  }
}

TEMPLATE_TEST_CASE(
    "Python-parity point cleaning preserves classes maps and ownership",
    "[cpp][clean][python-parity][points]", float, double) {
  auto input = duplicate_points<TestType>();
  const auto original = input.deep_copy();
  const auto result = tf::cpp::cleaned_points_with_map(input);

  REQUIRE((result.points.raw_shape() == tf::small_vector<int, 3>{4, 3}));
  REQUIRE(result.point_map.f.length() == 6);
  REQUIRE(result.point_map.kept_ids.length() == 4);
  CHECK(result.point_map.f[0] == result.point_map.f[2]);
  CHECK(result.point_map.f[3] == result.point_map.f[5]);
  CHECK(result.point_map.f[0] != result.point_map.f[1]);
  CHECK(result.point_map.f[1] != result.point_map.f[4]);

  for (std::size_t output_id = 0;
       output_id < result.point_map.kept_ids.length(); ++output_id) {
    const auto source_id = result.point_map.kept_ids[output_id];
    REQUIRE(source_id >= 0);
    REQUIRE(source_id < input.shape_at(0));
    CHECK(result.point_map.f[static_cast<std::size_t>(source_id)] ==
          static_cast<std::int32_t>(output_id));
    for (std::size_t coordinate = 0; coordinate < 3; ++coordinate)
      CHECK(result.points[output_id * 3 + coordinate] ==
            input[static_cast<std::size_t>(source_id) * 3 + coordinate]);
  }

  CHECK(same_array(input, original));

  input[0] = TestType{99};
  CHECK(result.points[0] != TestType{99});

  const auto repeated = tf::cpp::cleaned_points_with_map(original);
  CHECK(same_array(repeated.points, result.points));
  CHECK(same_array(repeated.point_map.f, result.point_map.f));
  CHECK(same_array(repeated.point_map.kept_ids, result.point_map.kept_ids));
}

TEMPLATE_TEST_CASE(
    "Python-parity tolerance cleaning has deterministic map classes",
    "[cpp][clean][python-parity][tolerance]", float, double) {
  auto input = make_array<TestType>(
      {0, 0, 0, TestType{0.001}, 0, 0, 1, 0, 0, 1, TestType{0.001}, 0, 2, 2, 2},
      {5, 3});
  const auto original = input.deep_copy();
  const auto result = tf::cpp::cleaned_points_with_map(input, TestType{0.01});

  CHECK((result.points.raw_shape() == tf::small_vector<int, 3>{3, 3}));
  CHECK(result.point_map.kept_ids.length() == 3);
  CHECK(result.point_map.f[0] == result.point_map.f[1]);
  CHECK(result.point_map.f[2] == result.point_map.f[3]);
  CHECK(result.point_map.f[0] != result.point_map.f[2]);
  CHECK(result.point_map.f[2] != result.point_map.f[4]);
  CHECK(same_array(input, original));

  const auto repeated = tf::cpp::cleaned_points_with_map(input, TestType{0.01});
  CHECK(same_array(repeated.points, result.points));
  CHECK(same_array(repeated.point_map.f, result.point_map.f));
  CHECK(same_array(repeated.point_map.kept_ids, result.point_map.kept_ids));
}

TEMPLATE_TEST_CASE(
    "Python-parity cleaning rejects negative tolerance and preserves zero",
    "[cpp][clean][python-parity][validation]", float, double) {
  const auto negative = TestType{-1};

  SECTION("every synchronous entry rejects finite negative tolerance") {
    auto points = near_duplicate_points<TestType>();
    auto soup = near_degenerate_soup<TestType>();
    auto mesh = near_degenerate_mesh<TestType>();

    CHECK_THROWS_AS(tf::cpp::cleaned_polygon_soup(soup, negative),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::cleaned_mesh(mesh.mesh(), negative),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::cleaned_mesh_with_maps(mesh.mesh(), negative),
                    std::invalid_argument);
    CHECK_THROWS_MATCHES(
        tf::cpp::cleaned_points(points, negative), std::invalid_argument,
        Catch::Matchers::Message("clean: tolerance must be non-negative"));
    CHECK_THROWS_AS(tf::cpp::cleaned_points_with_map(points, negative),
                    std::invalid_argument);
  }

  SECTION("an asynchronous entry propagates negative tolerance rejection") {
    auto mesh = near_degenerate_mesh<TestType>();
    CHECK_THROWS_AS(
        tf::cpp::async::cleaned_mesh_with_maps(mesh.mesh(), negative).get(),
        std::invalid_argument);
  }

  SECTION("zero keeps exact cleaning for every synchronous entry") {
    auto points = near_duplicate_points<TestType>();
    auto soup = near_degenerate_soup<TestType>();
    auto mesh = near_degenerate_mesh<TestType>();

    CHECK(tf::cpp::cleaned_polygon_soup(soup, TestType{}).size() == 1);
    CHECK(tf::cpp::cleaned_mesh(mesh.mesh(), TestType{}).size() == 1);
    CHECK(
        tf::cpp::cleaned_mesh_with_maps(mesh.mesh(), TestType{}).mesh.size() ==
        1);
    CHECK(tf::cpp::cleaned_points(points, TestType{}).shape_at(0) == 3);
    CHECK(tf::cpp::cleaned_points_with_map(points, TestType{})
              .point_map.kept_ids.length() == 3);
  }

  SECTION("every synchronous entry rejects nonfinite tolerance") {
    const auto nonfinite =
        std::array<TestType, 3>{std::numeric_limits<TestType>::quiet_NaN(),
                                std::numeric_limits<TestType>::infinity(),
                                -std::numeric_limits<TestType>::infinity()};
    for (const auto tolerance : nonfinite) {
      auto points = near_duplicate_points<TestType>();
      auto soup = near_degenerate_soup<TestType>();
      auto mesh = near_degenerate_mesh<TestType>();

      CHECK_THROWS_AS(tf::cpp::cleaned_polygon_soup(soup, tolerance),
                      std::invalid_argument);
      CHECK_THROWS_AS(tf::cpp::cleaned_mesh(mesh.mesh(), tolerance),
                      std::invalid_argument);
      CHECK_THROWS_AS(tf::cpp::cleaned_mesh_with_maps(mesh.mesh(), tolerance),
                      std::invalid_argument);
      CHECK_THROWS_AS(tf::cpp::cleaned_points(points, tolerance),
                      std::invalid_argument);
      CHECK_THROWS_AS(tf::cpp::cleaned_points_with_map(points, tolerance),
                      std::invalid_argument);
    }
  }
}

TEST_CASE("Python-parity point cleaning validates unsupported public shapes",
          "[cpp][clean][python-parity][validation]") {
  const auto malformed = make_array<float>({0, 1}, {2});
  CHECK_THROWS_AS(tf::cpp::cleaned_points(malformed), std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::cleaned_points_with_map(malformed),
                  std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "Python-parity mesh cleaning applies owning maps without mutation",
    "[cpp][clean][python-parity][mesh]", float, double) {
  auto input = duplicate_mesh<TestType>();
  const auto original_faces = tf::cpp::test::face_indices_of(input.polygons);
  const auto original_points = points_array(input.polygons.points_buffer());
  auto result = tf::cpp::cleaned_mesh_with_maps(input.mesh());
  const auto result_faces = tf::cpp::test::face_indices_of(result.mesh);

  REQUIRE(result.mesh.size() == 2);
  REQUIRE(result.mesh.points_buffer().size() == 4);
  REQUIRE(result.face_map.f.length() == 2);
  REQUIRE(result.face_map.kept_ids.length() == 2);
  REQUIRE(result.point_map.f.length() == 5);
  REQUIRE(result.point_map.kept_ids.length() == 4);
  CHECK(result.point_map.f[0] == result.point_map.f[3]);

  for (std::size_t source_face = 0; source_face < 2; ++source_face) {
    const auto output_face = result.face_map.f[source_face];
    REQUIRE(output_face >= 0);
    REQUIRE(static_cast<std::size_t>(output_face) < result.mesh.size());
    for (std::size_t corner = 0; corner < 3; ++corner) {
      const auto source_point = original_faces[source_face * 3 + corner];
      CHECK(result_faces[static_cast<std::size_t>(output_face) * 3 + corner] ==
            result.point_map.f[static_cast<std::size_t>(source_point)]);
    }
  }

  CHECK(same_array(tf::cpp::test::face_indices_of(input.polygons),
                   original_faces));
  CHECK(same_array(points_array(input.polygons.points_buffer()),
                   original_points));

  result.mesh.points_buffer().data_buffer()[0] = TestType{77};
  CHECK(input.polygons.points_buffer().data_buffer()[0] == TestType{0});
}

TEMPLATE_LIST_TEST_CASE(
    "Python EdgeMesh and fixed Mesh cleaning preserves every carrier axis",
    "[cpp][clean][python-parity][edge-mesh][mesh][matrix]", form_matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using mesh_result = tf::cpp::cleaned_mesh_result<Index, Real, TestType::dims>;
  using edge_result =
      tf::cpp::cleaned_edge_mesh_result<Index, Real, TestType::dims>;

  auto edge_input = python_edge_mesh<TestType>();
  auto mesh_input = python_fixed_mesh<TestType>();
  const auto original_edges = edges_array(edge_input.segments);
  const auto original_edge_points =
      points_array(edge_input.segments.points_buffer());
  const auto original_faces =
      tf::cpp::test::face_indices_of(mesh_input.polygons);
  const auto original_mesh_points =
      points_array(mesh_input.polygons.points_buffer());
  edge_input.place(local_transform<TestType>());
  mesh_input.place(local_transform<TestType>());

  const auto edge =
      tf::cpp::cleaned_edge_mesh_with_maps<Index, Real, TestType::dims>(
          edge_input.edge_mesh());
  const auto edge_direct =
      tf::cpp::cleaned_edge_mesh<Index, Real, TestType::dims>(
          edge_input.edge_mesh());
  const auto fixed =
      tf::cpp::cleaned_mesh_with_maps<Index, Real, TestType::dims, 3>(
          mesh_input.mesh());
  const auto fixed_direct =
      tf::cpp::cleaned_mesh<Index, Real, TestType::dims, 3>(mesh_input.mesh());

  STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(edge)>, edge_result>);
  STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(fixed)>, mesh_result>);
  STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(edge.edge_map.f)>,
                                tf::cpp::nd_array<Index>>);
  STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(fixed.face_map.f)>,
                                tf::cpp::nd_array<Index>>);
  STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(edge.edge_mesh)>,
                                typename TestType::edge_result_type>);
  STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(fixed.mesh)>,
                                typename TestType::mesh_result_type>);

  REQUIRE(edge.edge_mesh.size() == 2);
  REQUIRE(edge.edge_mesh.points_buffer().size() == 3);
  CHECK(edge.edge_map.f.length() == 2);
  CHECK(edge.edge_map.kept_ids.length() == 2);
  CHECK(edge.point_map.f.length() == 4);
  CHECK(edge.point_map.kept_ids.length() == 3);
  CHECK(edge.point_map.f[0] == edge.point_map.f[2]);
  CHECK(same_array(edges_array(edge_direct), edges_array(edge.edge_mesh)));
  CHECK(same_array(points_array(edge_direct.points_buffer()),
                   points_array(edge.edge_mesh.points_buffer())));

  REQUIRE(fixed.mesh.size() == 2);
  REQUIRE(fixed.mesh.points_buffer().size() == 4);
  CHECK(fixed.face_map.f.length() == 2);
  CHECK(fixed.face_map.kept_ids.length() == 2);
  CHECK(fixed.point_map.f.length() == 5);
  CHECK(fixed.point_map.kept_ids.length() == 4);
  CHECK(fixed.point_map.f[0] == fixed.point_map.f[3]);
  CHECK(same_array(tf::cpp::test::face_indices_of(fixed_direct),
                   tf::cpp::test::face_indices_of(fixed.mesh)));
  CHECK(same_array(points_array(fixed_direct.points_buffer()),
                   points_array(fixed.mesh.points_buffer())));

  CHECK(same_array(edges_array(edge_input.segments), original_edges));
  CHECK(same_array(points_array(edge_input.segments.points_buffer()),
                   original_edge_points));
  CHECK(same_array(tf::cpp::test::face_indices_of(mesh_input.polygons),
                   original_faces));
  CHECK(same_array(points_array(mesh_input.polygons.points_buffer()),
                   original_mesh_points));
}

TEMPLATE_LIST_TEST_CASE(
    "Python dynamic Mesh cleaning preserves jagged layout and typed maps",
    "[cpp][clean][python-parity][mesh][dynamic][matrix]", form_matrix_rows) {
  using Index = typename TestType::index_type;

  auto input = python_dynamic_mesh<TestType>();
  const auto original_indices = tf::cpp::test::face_indices_of(input.polygons);
  const auto original_offsets = tf::cpp::test::face_offsets_of(input.polygons);
  input.place(local_transform<TestType>());
  const auto result = tf::cpp::cleaned_mesh_with_maps(input.mesh());
  const auto direct = tf::cpp::cleaned_mesh(input.mesh());

  STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(result.mesh)>,
                                typename TestType::dynamic_mesh_result_type>);
  REQUIRE(result.mesh.size() == 2);
  REQUIRE(result.mesh.points_buffer().size() == 5);
  REQUIRE(tf::cpp::test::face_offsets_of(result.mesh).length() == 3);
  CHECK(tf::cpp::test::face_offsets_of(result.mesh)[0] == Index{0});
  CHECK(tf::cpp::test::face_offsets_of(result.mesh)[1] == Index{3});
  CHECK(tf::cpp::test::face_offsets_of(result.mesh)[2] == Index{7});
  CHECK(result.face_map.f.length() == 2);
  CHECK(result.face_map.kept_ids.length() == 2);
  CHECK(result.point_map.f.length() == 6);
  CHECK(result.point_map.kept_ids.length() == 5);
  CHECK(result.point_map.f[0] == result.point_map.f[3]);
  CHECK(same_array(tf::cpp::test::face_offsets_of(direct),
                   tf::cpp::test::face_offsets_of(result.mesh)));
  CHECK(same_array(tf::cpp::test::face_indices_of(direct),
                   tf::cpp::test::face_indices_of(result.mesh)));
  CHECK(same_array(original_offsets,
                   tf::cpp::test::face_offsets_of(input.polygons)));
  CHECK(same_array(original_indices,
                   tf::cpp::test::face_indices_of(input.polygons)));
}

TEMPLATE_LIST_TEST_CASE(
    "Typed cleaning removes isolated degenerate and duplicate primitives",
    "[cpp][clean][python-parity][cleanup][matrix]", form_matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;

  auto edges = cleanup_edge_mesh<TestType>();
  const auto cleaned_edges =
      tf::cpp::cleaned_edge_mesh_with_maps<Index, Real, TestType::dims>(
          edges.edge_mesh());
  CHECK(cleaned_edges.edge_mesh.size() == 2);
  CHECK(cleaned_edges.edge_mesh.points_buffer().size() == 3);
  CHECK(cleaned_edges.edge_map.f.length() == 4);
  CHECK(cleaned_edges.edge_map.kept_ids.length() == 2);
  CHECK(cleaned_edges.edge_map.f[0] == cleaned_edges.edge_map.f[1]);
  CHECK(cleaned_edges.edge_map.f[2] == Index{4});
  CHECK(cleaned_edges.point_map.kept_ids.length() == 3);

  const auto kept_duplicate_edges =
      tf::cpp::cleaned_edge_mesh<Index, Real, TestType::dims>(
          edges.edge_mesh(), Real{}, false, true);
  CHECK(kept_duplicate_edges.size() == 3);
  CHECK(kept_duplicate_edges.points_buffer().size() == 3);

  auto mesh = cleanup_fixed_mesh<TestType>();
  const auto cleaned =
      tf::cpp::cleaned_mesh_with_maps<Index, Real, TestType::dims, 3>(
          mesh.mesh());
  CHECK(cleaned.mesh.size() == 1);
  CHECK(cleaned.mesh.points_buffer().size() == 3);
  CHECK(cleaned.face_map.f.length() == 3);
  CHECK(cleaned.face_map.kept_ids.length() == 1);
  CHECK(((cleaned.face_map.f[0] == Index{0} &&
          cleaned.face_map.f[1] == Index{3}) ||
         (cleaned.face_map.f[1] == Index{0} &&
          cleaned.face_map.f[0] == Index{3})));
  CHECK(cleaned.face_map.f[2] == Index{3});

  const auto kept_duplicate_faces =
      tf::cpp::cleaned_mesh<Index, Real, TestType::dims, 3>(mesh.mesh(), Real{},
                                                            false, true);
  CHECK(kept_duplicate_faces.size() == 2);
  CHECK(kept_duplicate_faces.points_buffer().size() == 3);

  const auto near_points = [&]() -> std::vector<Real> {
    if constexpr (TestType::dims == 2)
      return {0, 0, Real{0.001}, Real{0.001}, 1, 0};
    else
      return {0, 0, 0, Real{0.001}, Real{0.001}, Real{0.001}, 1, 0, 0};
  }();
  typename TestType::mesh_type near{
      tf::cpp::test::polygons_of<Index, Real, TestType::dims>(
          std::vector<Index>{0, 1, 2}, near_points)};
  CHECK(tf::cpp::cleaned_mesh<Index, Real, TestType::dims, 3>(near.mesh())
            .size() == 1);
  CHECK(tf::cpp::cleaned_mesh<Index, Real, TestType::dims, 3>(near.mesh(),
                                                              Real{0.01})
            .size() == 0);
}

TEMPLATE_LIST_TEST_CASE(
    "Python polygon-soup fixtures clean typed segment and triangle arrays",
    "[cpp][clean][python-parity][polygon-soup][matrix]", form_matrix_rows) {
  check_python_polygon_soup_fixture<TestType, 2>();
  check_python_polygon_soup_fixture<TestType, 3>();
}

TEMPLATE_LIST_TEST_CASE(
    "Typed polygon soups validate empty singleton and async ownership",
    "[cpp][clean][python-parity][polygon-soup][empty][validation][async]["
    "matrix]",
    form_matrix_rows) {
  check_polygon_soup_boundaries<TestType, 2>();
  check_polygon_soup_boundaries<TestType, 3>();
}

TEMPLATE_LIST_TEST_CASE(
    "Typed clean handles empty malformed and asynchronous form snapshots",
    "[cpp][clean][python-parity][empty][validation][async][matrix]",
    form_matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  using mesh_result = tf::cpp::cleaned_mesh_result<Index, Real, TestType::dims,
                                                   tf::dynamic_size>;
  using edge_result =
      tf::cpp::cleaned_edge_mesh_result<Index, Real, TestType::dims>;

  auto dynamic = empty_dynamic_mesh<TestType>();
  const auto dynamic_kept =
      tf::cpp::cleaned_mesh_with_maps<Index, Real, TestType::dims,
                                      tf::dynamic_size>(dynamic.mesh(), Real{},
                                                        true, false);
  CHECK(dynamic_kept.mesh.size() == 0);
  CHECK(dynamic_kept.mesh.points_buffer().size() == 2);
  CHECK(dynamic_kept.face_map.f.length() == 0);
  CHECK(dynamic_kept.point_map.f.length() == 3);
  CHECK(dynamic_kept.point_map.kept_ids.length() == 2);

  const auto dynamic_removed =
      tf::cpp::cleaned_mesh_with_maps<Index, Real, TestType::dims,
                                      tf::dynamic_size>(dynamic.mesh());
  CHECK(dynamic_removed.mesh.points_buffer().size() == 0);
  CHECK(dynamic_removed.point_map.f.length() == 3);
  CHECK(dynamic_removed.point_map.kept_ids.length() == 0);
  for (const auto mapped : dynamic_removed.point_map.f)
    CHECK(mapped == Index{3});

  const auto three_points = [&]() -> std::vector<Real> {
    if constexpr (TestType::dims == 2)
      return {0, 0, 1, 0, 0, 1};
    else
      return {0, 0, 0, 1, 0, 0, 0, 1, 0};
  }();
  typename TestType::mesh_type fixed{
      tf::cpp::test::polygons_of<Index, Real, TestType::dims>(
          std::vector<Index>{}, three_points)};
  STATIC_REQUIRE(std::is_same_v<
                 std::decay_t<decltype(tf::cpp::cleaned_mesh(fixed.mesh()))>,
                 typename TestType::mesh_result_type>);
  CHECK(tf::cpp::cleaned_mesh(fixed.mesh()).size() == 0);
  typename TestType::edge_mesh_type empty_edges{
      tf::cpp::test::segments_of<Index, Real, TestType::dims>(
          std::vector<Index>{}, three_points)};
  const auto edge_kept =
      tf::cpp::cleaned_edge_mesh_with_maps<Index, Real, TestType::dims>(
          empty_edges.edge_mesh(), Real{}, true, false);
  CHECK(edge_kept.edge_mesh.size() == 0);
  CHECK(edge_kept.edge_mesh.points_buffer().size() == 3);
  CHECK(edge_kept.edge_map.f.length() == 0);
  CHECK(edge_kept.point_map.f.length() == 3);

  // a carrier over empty storage is the EMPTY one, which cleans to itself
  typename TestType::mesh_type empty_mesh;
  typename TestType::edge_mesh_type empty_edge_mesh;
  CHECK(tf::cpp::cleaned_mesh(empty_mesh.mesh()).size() == 0);
  CHECK(tf::cpp::cleaned_edge_mesh(empty_edge_mesh.edge_mesh()).size() == 0);

  // the cache owns the indices for the reading it answers, so a read refuses
  // every corner that names a point the geometry does not have
  typename TestType::mesh_type beyond{
      tf::cpp::test::polygons_of<Index, Real, TestType::dims>(
          std::vector<Index>{0, 1, 3}, three_points)};
  CHECK_THROWS_AS(tf::cpp::cleaned_mesh(beyond.mesh()), std::out_of_range);
  typename TestType::mesh_type bad_mesh{
      tf::cpp::test::polygons_of<Index, Real, TestType::dims>(
          std::vector<Index>{0, 1, 2}, three_points)};
  bad_mesh.polygons.points_buffer().data_buffer().allocate(2 * TestType::dims);
  bad_mesh.cache.points_changed();
  typename TestType::edge_mesh_type bad_edges{
      tf::cpp::test::segments_of<Index, Real, TestType::dims>(
          std::vector<Index>{0, 2}, three_points)};
  bad_edges.segments.points_buffer().data_buffer().allocate(2 * TestType::dims);
  bad_edges.cache.points_changed();
  CHECK_THROWS_AS(tf::cpp::cleaned_mesh(bad_mesh.mesh()), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::cleaned_edge_mesh(bad_edges.edge_mesh()),
                  std::out_of_range);
  CHECK_THROWS_AS((tf::cpp::cleaned_mesh<Index, Real, TestType::dims, 3>(
                      fixed.mesh(), std::numeric_limits<Real>::infinity())),
                  std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::cleaned_edge_mesh<Index, Real, TestType::dims>(
                      empty_edges.edge_mesh(), Real{-1})),
                  std::invalid_argument);

  auto async_mesh_input = python_dynamic_mesh<TestType>();
  auto async_edge_input = python_edge_mesh<TestType>();
  const auto expected_mesh =
      tf::cpp::cleaned_mesh_with_maps(async_mesh_input.mesh());
  const auto expected_edge =
      tf::cpp::cleaned_edge_mesh_with_maps<Index, Real, TestType::dims>(
          async_edge_input.edge_mesh());
  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto pending_mesh = tf::cpp::async::cleaned_mesh_with_maps(
      mutating_resolver{submissions, [] {}}, async_mesh_input.mesh());
  auto pending_edge =
      tf::cpp::async::cleaned_edge_mesh_with_maps<Index, Real, TestType::dims>(
          mutating_resolver{submissions, [] {}}, async_edge_input.edge_mesh());
  STATIC_REQUIRE(
      std::is_same_v<decltype(pending_mesh), std::future<mesh_result>>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(pending_edge), std::future<edge_result>>);
  const auto async_mesh = pending_mesh.get();
  const auto async_edge = pending_edge.get();
  CHECK(same_array(tf::cpp::test::face_indices_of(async_mesh.mesh),
                   tf::cpp::test::face_indices_of(expected_mesh.mesh)));
  CHECK(same_array(async_mesh.point_map.f, expected_mesh.point_map.f));
  CHECK(same_array(edges_array(async_edge.edge_mesh),
                   edges_array(expected_edge.edge_mesh)));
  CHECK(same_array(async_edge.point_map.f, expected_edge.point_map.f));
  CHECK(submissions->load(std::memory_order_relaxed) == 2);
}

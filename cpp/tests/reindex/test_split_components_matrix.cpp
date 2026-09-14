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
#include "trueform/cpp/reindex.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

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

template <typename Index, typename Real, std::size_t Dims> struct matrix_row {
  using real_type = Real;
  using index_type = Index;
  static constexpr auto dims = Dims;
  using mesh_type = tf::cpp::test::owned_mesh<Index, Real, Dims>;
  using edge_mesh_type = tf::cpp::test::owned_edge_mesh<Index, Real, Dims>;
  using mesh_result_type = tf::polygons_buffer<Index, Real, Dims, 3>;
  using dynamic_mesh_result_type =
      tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size>;
  using edge_result_type = tf::segments_buffer<Index, Real, Dims>;
};

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

using matrix_rows = std::tuple<
    matrix_row<std::int32_t, float, 2>, matrix_row<std::int32_t, float, 3>,
    matrix_row<std::int64_t, float, 2>, matrix_row<std::int64_t, float, 3>,
    matrix_row<std::int32_t, double, 2>, matrix_row<std::int32_t, double, 3>,
    matrix_row<std::int64_t, double, 2>, matrix_row<std::int64_t, double, 3>>;

template <typename T>
auto make_array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(values.size());
  std::copy(values.begin(), values.end(), buffer.begin());
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename T>
auto same_array(const tf::cpp::nd_array<T> &first,
                const tf::cpp::nd_array<T> &second) -> bool {
  if (first.raw_shape() != second.raw_shape() ||
      first.length() != second.length())
    return false;
  return std::equal(first.begin(), first.end(), second.begin());
}

template <typename Index>
auto same_map(const tf::cpp::index_map<Index> &first,
              const tf::cpp::index_map<Index> &second) -> bool {
  return same_array(first.f, second.f) &&
         same_array(first.kept_ids, second.kept_ids);
}

template <typename T>
auto has_values(const tf::cpp::nd_array<T> &value,
                std::initializer_list<T> expected) -> bool {
  return value.length() == expected.size() &&
         std::equal(value.begin(), value.end(), expected.begin());
}

template <typename T>
auto has_selected_points(const tf::cpp::nd_array<T> &actual,
                         const tf::cpp::nd_array<T> &source,
                         std::initializer_list<int> point_ids) -> bool {
  if (actual.ndim() != 2 || source.ndim() != 2 ||
      actual.shape_at(0) != static_cast<int>(point_ids.size()) ||
      actual.shape_at(1) != source.shape_at(1))
    return false;
  const auto dims = static_cast<std::size_t>(source.shape_at(1));
  std::size_t output = 0;
  for (const auto point_id : point_ids)
    for (std::size_t dim = 0; dim < dims; ++dim)
      if (actual[output++] !=
          source[static_cast<std::size_t>(point_id) * dims + dim])
        return false;
  return true;
}

template <typename Row>
auto python_points() -> tf::cpp::nd_array<typename Row::real_type> {
  using Real = typename Row::real_type;
  constexpr auto Dims = Row::dims;
  constexpr double coordinates[6][3] = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                                        {0.5, 1.0, 0.0}, {2.0, 0.0, 0.0},
                                        {2.5, 1.0, 0.0}, {3.0, 0.0, 0.0}};
  tf::buffer<Real> buffer;
  buffer.allocate(6 * Dims);
  for (std::size_t point = 0; point < 6; ++point)
    for (std::size_t dim = 0; dim < Dims; ++dim)
      buffer[point * Dims + dim] = static_cast<Real>(coordinates[point][dim]);
  return tf::cpp::nd_array<Real>::from_buffer(std::move(buffer),
                                              {6, static_cast<int>(Dims)});
}

template <typename Row> auto python_coordinates() {
  using Real = typename Row::real_type;
  constexpr auto Dims = Row::dims;
  constexpr double coordinates[6][3] = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                                        {0.5, 1.0, 0.0}, {2.0, 0.0, 0.0},
                                        {2.5, 1.0, 0.0}, {3.0, 0.0, 0.0}};
  std::vector<Real> values(6 * Dims);
  for (std::size_t point = 0; point < 6; ++point)
    for (std::size_t dim = 0; dim < Dims; ++dim)
      values[point * Dims + dim] = static_cast<Real>(coordinates[point][dim]);
  return values;
}

template <typename Row> auto fixed_python_mesh() -> typename Row::mesh_type {
  using Index = typename Row::index_type;
  using Real = typename Row::real_type;
  return {tf::cpp::test::polygons_of<Index, Real, Row::dims>(
      std::vector<Index>{0, 1, 2, 1, 3, 2, 3, 4, 5, 3, 5, 4},
      python_coordinates<Row>())};
}

template <typename Row>
auto dynamic_python_mesh()
    -> tf::cpp::test::mixed_mesh_of<typename Row::mesh_type> {
  using Index = typename Row::index_type;
  using Real = typename Row::real_type;
  return {tf::cpp::test::polygons_of<Index, Real, Row::dims>(
      std::vector<Index>{0, 3, 6, 9, 12},
      std::vector<Index>{0, 1, 2, 1, 3, 2, 3, 4, 5, 3, 5, 4},
      python_coordinates<Row>())};
}

template <typename Row> auto python_edge_coordinates() {
  using Real = typename Row::real_type;
  constexpr auto Dims = Row::dims;
  constexpr double coordinates[5][3] = {{0.0, 0.0, 0.0},
                                        {1.0, 0.0, 0.0},
                                        {1.0, 1.0, 0.0},
                                        {2.0, 0.0, 0.0},
                                        {3.0, 0.0, 0.0}};
  std::vector<Real> values(5 * Dims);
  for (std::size_t point = 0; point < 5; ++point)
    for (std::size_t dim = 0; dim < Dims; ++dim)
      values[point * Dims + dim] = static_cast<Real>(coordinates[point][dim]);
  return values;
}

template <typename Row>
auto python_edge_mesh() -> typename Row::edge_mesh_type {
  using Index = typename Row::index_type;
  using Real = typename Row::real_type;
  return {tf::cpp::test::segments_of<Index, Real, Row::dims>(
      std::vector<Index>{0, 1, 1, 2, 3, 4}, python_edge_coordinates<Row>())};
}

template <typename Row> auto translation() {
  using Real = typename Row::real_type;
  constexpr auto Width = Row::dims + 1;
  std::array<Real, Width * Width> values{};
  for (std::size_t index = 0; index < Width; ++index)
    values[index * Width + index] = Real{1};
  values[Row::dims] = Real{100};
  return values;
}

template <typename Row, std::size_t Vertices, typename = void>
struct has_fixed_ids_selector : std::false_type {};

template <typename Row, std::size_t Vertices>
struct has_fixed_ids_selector<
    Row, Vertices,
    std::void_t<decltype(tf::cpp::reindexed_by_ids<typename Row::index_type,
                                                   typename Row::real_type,
                                                   Row::dims, Vertices>(
        std::declval<const tf::cpp::nd_array<typename Row::index_type> &>(),
        std::declval<const tf::cpp::nd_array<typename Row::real_type> &>(),
        std::declval<const tf::cpp::nd_array<typename Row::index_type> &>()))>>
    : std::true_type {};

template <typename Row, std::size_t Vertices, typename = void>
struct has_async_fixed_ids_selector : std::false_type {};

template <typename Row, std::size_t Vertices>
struct has_async_fixed_ids_selector<
    Row, Vertices,
    std::void_t<decltype(tf::cpp::async::reindexed_by_ids<
                         typename Row::index_type, typename Row::real_type,
                         Row::dims, Vertices>(
        std::declval<const tf::cpp::nd_array<typename Row::index_type> &>(),
        std::declval<const tf::cpp::nd_array<typename Row::real_type> &>(),
        std::declval<const tf::cpp::nd_array<typename Row::index_type> &>()))>>
    : std::true_type {};

using fixed_selector_detection_row = matrix_row<std::int32_t, float, 3>;
static_assert(has_fixed_ids_selector<fixed_selector_detection_row, 2>::value);
static_assert(has_fixed_ids_selector<fixed_selector_detection_row, 3>::value);
static_assert(!has_fixed_ids_selector<fixed_selector_detection_row, 4>::value);
static_assert(
    has_async_fixed_ids_selector<fixed_selector_detection_row, 2>::value);
static_assert(
    has_async_fixed_ids_selector<fixed_selector_detection_row, 3>::value);
static_assert(
    !has_async_fixed_ids_selector<fixed_selector_detection_row, 4>::value);

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
auto check_split_components_case(const tf::cpp::nd_array<std::int32_t> &labels)
    -> void {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  constexpr auto Dims = Row::dims;
  using result_type = tf::cpp::split_components_result<
      tf::polygons_buffer<Index, Real, Dims, Ngon>>;
  auto input = [] {
    if constexpr (Ngon == 3)
      return fixed_python_mesh<Row>();
    else
      return dynamic_python_mesh<Row>();
  }();
  input.place(translation<Row>());
  const auto original_points = points_array(input.polygons.points_buffer());
  const auto original_indices = tf::cpp::test::face_indices_of(input.polygons);
  const auto original_offsets = tf::cpp::test::face_offsets_of(input.polygons);

  auto result = tf::cpp::split_into_components(input.mesh(), labels);
  static_assert(std::is_same_v<decltype(result), result_type>);
  REQUIRE(result.components.size() == 2);
  CHECK(has_values(result.labels, {std::int32_t{0}, std::int32_t{1}}));
  for (const auto &component : result.components) {
    CHECK(component.size() == 2);
    CHECK(component.points_buffer().size() >= 3);
    CHECK(component.points_buffer().data_buffer()[0] < Real{100});
  }

  CHECK(
      has_values(tf::cpp::test::face_indices_of(result.components[0]),
                 {Index{0}, Index{1}, Index{2}, Index{1}, Index{3}, Index{2}}));
  CHECK(
      has_values(tf::cpp::test::face_indices_of(result.components[1]),
                 {Index{0}, Index{1}, Index{2}, Index{0}, Index{2}, Index{1}}));
  CHECK(result.components[0].points_buffer().size() == 4);
  CHECK(result.components[1].points_buffer().size() == 3);
  CHECK(has_selected_points(points_array(result.components[0].points_buffer()),
                            original_points, {0, 1, 2, 3}));
  CHECK(has_selected_points(points_array(result.components[1].points_buffer()),
                            original_points, {3, 4, 5}));
  if constexpr (Ngon != 3) {
    CHECK(has_values(tf::cpp::test::face_offsets_of(result.components[0]),
                     {Index{0}, Index{3}, Index{6}}));
    CHECK(has_values(tf::cpp::test::face_offsets_of(result.components[1]),
                     {Index{0}, Index{3}, Index{6}}));
  } else {
    CHECK(tf::cpp::test::face_indices_of(result.components[0]).length() == 6);
  }

  CHECK(same_array(points_array(input.polygons.points_buffer()),
                   original_points));
  CHECK(same_array(tf::cpp::test::face_indices_of(input.polygons),
                   original_indices));
  if constexpr (Ngon != 3)
    CHECK(same_array(tf::cpp::test::face_offsets_of(input.polygons),
                     original_offsets));

  const auto explicit_labels = make_array<std::int32_t>({2, 0, 1, 1}, {4});
  const auto explicitly_split =
      tf::cpp::split_into_components(input.mesh(), explicit_labels);
  REQUIRE(explicitly_split.components.size() == 3);
  CHECK(has_values(explicitly_split.labels,
                   {std::int32_t{0}, std::int32_t{1}, std::int32_t{2}}));
  CHECK(explicitly_split.components[0].size() == 1);
  CHECK(explicitly_split.components[1].size() == 2);
  CHECK(explicitly_split.components[2].size() == 1);

  auto deduced_split = tf::cpp::split_into_components(input.mesh(), labels);
  static_assert(std::is_same_v<decltype(deduced_split), result_type>);
  CHECK(deduced_split.components.size() == 2);
  auto default_pending =
      tf::cpp::async::split_into_components(input.mesh(), labels);
  static_assert(
      std::is_same_v<decltype(default_pending), std::future<result_type>>);
  CHECK(default_pending.get().components.size() == 2);
  CHECK(result.components[0].points_buffer().size() == 4);
}

template <typename Row, std::size_t Ngon>
auto check_reindex_selector_case(
    const tf::cpp::nd_array<typename Row::index_type> &face_ids,
    const tf::cpp::nd_array<std::int8_t> &face_mask,
    const tf::cpp::nd_array<typename Row::index_type> &kept_point_ids,
    const tf::cpp::nd_array<std::int8_t> &kept_point_mask) -> void {
  using Index = typename Row::index_type;
  auto input = [] {
    if constexpr (Ngon == 3)
      return fixed_python_mesh<Row>();
    else
      return dynamic_python_mesh<Row>();
  }();
  const auto original_points = points_array(input.polygons.points_buffer());
  const auto original_indices = tf::cpp::test::face_indices_of(input.polygons);

  const auto by_ids =
      tf::cpp::reindexed_by_ids_with_maps(input.mesh(), face_ids);
  const auto by_mask =
      tf::cpp::reindexed_by_mask_with_maps(input.mesh(), face_mask);
  const auto by_point_ids = tf::cpp::reindexed_by_ids_on_points_with_maps(
      input.mesh(), kept_point_ids);
  const auto by_point_mask = tf::cpp::reindexed_by_mask_on_points_with_maps(
      input.mesh(), kept_point_mask);

  const auto reapplied =
      tf::cpp::reindexed(input.mesh(), by_ids.face_map, by_ids.point_map);
  CHECK(same_array(tf::cpp::test::face_indices_of(reapplied),
                   tf::cpp::test::face_indices_of(by_ids.mesh)));
  CHECK(same_array(points_array(reapplied.points_buffer()),
                   points_array(by_ids.mesh.points_buffer())));
  CHECK(by_ids.mesh.size() == 2);
  CHECK(has_values(by_ids.face_map.kept_ids, {Index{2}, Index{0}}));
  CHECK(has_values(by_mask.face_map.kept_ids, {Index{0}, Index{2}}));
  CHECK(by_point_ids.mesh.size() == 2);
  CHECK(by_point_mask.mesh.size() == 2);
  CHECK(has_values(by_point_ids.point_map.kept_ids,
                   {Index{3}, Index{4}, Index{5}}));
  CHECK(same_map(by_point_ids.face_map, by_point_mask.face_map));
  CHECK(same_map(by_point_ids.point_map, by_point_mask.point_map));
  CHECK(same_array(tf::cpp::test::face_indices_of(by_point_ids.mesh),
                   tf::cpp::test::face_indices_of(by_point_mask.mesh)));
  CHECK(same_array(points_array(by_point_ids.mesh.points_buffer()),
                   points_array(by_point_mask.mesh.points_buffer())));
  if constexpr (Ngon != 3)
    CHECK(same_array(tf::cpp::test::face_offsets_of(by_point_ids.mesh),
                     tf::cpp::test::face_offsets_of(by_point_mask.mesh)));
  CHECK(same_array(points_array(input.polygons.points_buffer()),
                   original_points));
  CHECK(same_array(tf::cpp::test::face_indices_of(input.polygons),
                   original_indices));
}

template <typename Row, std::size_t Ngon>
auto check_split_domains_case() -> void {
  using Real = typename Row::real_type;
  using Index = typename Row::index_type;
  constexpr auto Dims = Row::dims;
  using result_type = tf::cpp::split_domains_result<Index, Real, Dims, Ngon>;
  auto input = [] {
    if constexpr (Ngon == 3)
      return fixed_python_mesh<Row>();
    else
      return dynamic_python_mesh<Row>();
  }();
  auto labels_array =
      make_array<Index>({Index{0}, Index{2}, Index{0}, Index{2}, Index{1},
                         Index{2}, Index{1}, Index{2}},
                        {4, 2});
  tf::cpp::domain_labels_result<Index> labels(labels_array, Index{2},
                                              Index{-1});
  const auto original_indices = tf::cpp::test::face_indices_of(input.polygons);
  auto result = tf::cpp::split_into_domains(input.mesh(), labels);
  static_assert(std::is_same_v<decltype(result), result_type>);
  REQUIRE(result.components.size() == 2);
  CHECK(has_values(result.labels, {Index{0}, Index{1}}));
  static_assert(std::is_same_v<typename decltype(result.components)::value_type,
                               tf::polygons_buffer<Index, Real, Dims, Ngon>>);
  CHECK(result.components[0].size() == 2);
  CHECK(result.components[1].size() == 2);
  CHECK(same_array(tf::cpp::test::face_indices_of(input.polygons),
                   original_indices));

  auto pending = tf::cpp::async::split_into_domains<Index, Real, Dims, Ngon>(
      input.mesh(), labels);
  labels_array[0] = Index{2};
  CHECK(pending.get().components.size() == 2);
}

} // namespace

TEMPLATE_LIST_TEST_CASE(
    "split components matches fixed and dynamic Python mesh fixtures",
    "[cpp][reindex][split-components][matrix][python-parity][mesh]",
    matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;
  using result_type =
      tf::cpp::split_components_result<typename TestType::mesh_result_type>;
  using function_type =
      result_type (*)(const tf::cpp::mesh<Index, Real, Dims> &,
                      const tf::cpp::nd_array<std::int32_t> &);
  const function_type symbol =
      &tf::cpp::split_into_components<Index, Real, Dims, 3>;
  REQUIRE(symbol != nullptr);

  const auto labels = make_array<std::int32_t>({0, 0, 1, 1}, {4});
  check_split_components_case<TestType, 3>(labels);
  check_split_components_case<TestType, tf::dynamic_size>(labels);
}

TEMPLATE_LIST_TEST_CASE(
    "split components matches Python edge mesh fixtures",
    "[cpp][reindex][split-components][matrix][python-parity][edge-mesh]",
    matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;
  using result_type =
      tf::cpp::split_components_result<typename TestType::edge_result_type>;
  using function_type =
      result_type (*)(const tf::cpp::edge_mesh<Index, Real, Dims> &,
                      const tf::cpp::nd_array<std::int32_t> &);
  const function_type symbol =
      &tf::cpp::split_into_components<Index, Real, Dims>;
  REQUIRE(symbol != nullptr);

  auto input = python_edge_mesh<TestType>();
  input.place(translation<TestType>());
  const auto original_edges = edges_array(input.segments);
  const auto original_points = points_array(input.segments.points_buffer());
  const auto labels = make_array<std::int32_t>({0, 0, 1}, {3});
  auto result = symbol(input.edge_mesh(), labels);

  REQUIRE(result.components.size() == 2);
  CHECK(has_values(result.labels, {std::int32_t{0}, std::int32_t{1}}));
  CHECK(has_values(edges_array(result.components[0]),
                   {Index{0}, Index{1}, Index{1}, Index{2}}));
  CHECK(has_values(edges_array(result.components[1]), {Index{0}, Index{1}}));
  CHECK(result.components[0].points_buffer().size() == 3);
  CHECK(result.components[1].points_buffer().size() == 2);
  CHECK(has_selected_points(points_array(result.components[0].points_buffer()),
                            original_points, {0, 1, 2}));
  CHECK(has_selected_points(points_array(result.components[1].points_buffer()),
                            original_points, {3, 4}));
  CHECK(result.components[0].points_buffer().data_buffer()[0] == Real{0});
  CHECK(same_array(edges_array(input.segments), original_edges));
  CHECK(same_array(points_array(input.segments.points_buffer()),
                   original_points));
}

TEMPLATE_LIST_TEST_CASE(
    "split components handles default-valued labels and every empty carrier",
    "[cpp][reindex][split-components][matrix][labels][empty]", matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  auto fixed = fixed_python_mesh<TestType>();
  const auto default_labels = make_array<std::int32_t>({0, 0, 0, 0}, {4});
  auto single = tf::cpp::split_into_components<Index, Real, Dims>(
      fixed.mesh(), default_labels);
  REQUIRE(single.components.size() == 1);
  CHECK(single.labels[0] == 0);
  CHECK(single.components[0].size() == 4);
  CHECK(single.components[0].points_buffer().size() == 6);

  typename TestType::mesh_type empty_fixed;
  tf::cpp::test::mixed_mesh_of<typename TestType::mesh_type> empty_dynamic;
  typename TestType::edge_mesh_type empty_edges;
  const auto no_labels = make_array<std::int32_t>({}, {0});

  const auto fixed_result = tf::cpp::split_into_components<Index, Real, Dims>(
      empty_fixed.mesh(), no_labels);
  const auto dynamic_result = tf::cpp::split_into_components<Index, Real, Dims>(
      empty_dynamic.mesh(), no_labels);
  const auto edge_result = tf::cpp::split_into_components<Index, Real, Dims>(
      empty_edges.edge_mesh(), no_labels);
  CHECK(fixed_result.components.empty());
  CHECK(dynamic_result.components.empty());
  CHECK(edge_result.components.empty());
  CHECK(fixed_result.labels.raw_shape() == tf::small_vector<int, 3>{0});
  CHECK(dynamic_result.labels.raw_shape() == tf::small_vector<int, 3>{0});
  CHECK(edge_result.labels.raw_shape() == tf::small_vector<int, 3>{0});
}

TEMPLATE_LIST_TEST_CASE(
    "split components retains typed mesh and edge labels past the caller's "
    "handle",
    "[cpp][reindex][split-components][matrix][async][ownership]", matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;
  auto submissions = std::make_shared<std::atomic<int>>(0);

  auto mesh = dynamic_python_mesh<TestType>();
  auto mesh_labels = make_array<std::int32_t>({0, 0, 1, 1}, {4});
  auto mesh_pending = tf::cpp::async::split_into_components<mutating_resolver,
                                                            Index, Real, Dims>(
      mutating_resolver{submissions, [&] { mesh_labels.destroy(); }},
      mesh.mesh(), mesh_labels);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  auto mesh_result = mesh_pending.get();
  REQUIRE(mesh_result.components.size() == 2);
  CHECK(mesh_result.components[0].size() == 2);

  auto edges = python_edge_mesh<TestType>();
  auto edge_labels = make_array<std::int32_t>({0, 0, 1}, {3});
  auto edge_pending = tf::cpp::async::split_into_components<mutating_resolver,
                                                            Index, Real, Dims>(
      mutating_resolver{submissions, [&] { edge_labels.destroy(); }},
      edges.edge_mesh(), edge_labels);
  CHECK(submissions->load(std::memory_order_relaxed) == 2);
  auto edge_result = edge_pending.get();
  REQUIRE(edge_result.components.size() == 2);
  CHECK(edge_result.components[0].size() == 2);
}

TEST_CASE(
    "split components validates labels and carrier indices before splitting",
    "[cpp][reindex][split-components][validation]") {
  using Row = matrix_row<std::int64_t, double, 2>;
  auto mesh = fixed_python_mesh<Row>();
  CHECK_THROWS_AS((tf::cpp::split_into_components<std::int64_t, double, 2>(
                      mesh.mesh(), tf::cpp::nd_array<std::int32_t>{})),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::split_into_components<std::int64_t, double, 2>(
          mesh.mesh(), make_array<std::int32_t>({0, 0, 1, 1}, {2, 2}))),
      std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::split_into_components<std::int64_t, double, 2>(
                      mesh.mesh(), make_array<std::int32_t>({0}, {1}))),
                  std::invalid_argument);

  // the cache owns the indices for the reading it answers, so a read refuses
  // every corner that names a point the geometry does not have
  auto invalid_faces = fixed_python_mesh<Row>();
  invalid_faces.polygons.faces_buffer().data_buffer()[0] = 6;
  invalid_faces.cache.faces_changed();
  CHECK_THROWS_AS(
      (tf::cpp::split_into_components<std::int64_t, double, 2>(
          invalid_faces.mesh(), make_array<std::int32_t>({0, 0, 1, 1}, {4}))),
      std::out_of_range);

  auto invalid_edges = python_edge_mesh<Row>();
  invalid_edges.segments.edges_buffer().data_buffer()[0] = -1;
  invalid_edges.cache.edges_changed();
  CHECK_THROWS_AS(
      (tf::cpp::split_into_components<std::int64_t, double, 2>(
          invalid_edges.edge_mesh(), make_array<std::int32_t>({0, 0, 1}, {3}))),
      std::out_of_range);

  auto invalid_dynamic = dynamic_python_mesh<Row>();
  invalid_dynamic.polygons.points_buffer().data_buffer().allocate(0);
  invalid_dynamic.cache.points_changed();
  CHECK_THROWS_AS(
      (tf::cpp::split_into_components<std::int64_t, double, 2>(
          invalid_dynamic.mesh(), make_array<std::int32_t>({0, 0, 0, 0}, {4}))),
      std::out_of_range);

  mesh = {};
  CHECK_THROWS_AS(
      (tf::cpp::split_into_components<std::int64_t, double, 2>(
          mesh.mesh(), make_array<std::int32_t>({0, 0, 1, 1}, {4}))),
      std::invalid_argument);
}

TEMPLATE_LIST_TEST_CASE(
    "typed selectors close raw point mesh dynamic and edge Python parity",
    "[cpp][reindex][selectors][matrix][python-parity]", matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  const auto points = python_points<TestType>();
  const auto point_ids = make_array<Index>({Index{4}, Index{1}}, {2});
  const auto selected_points =
      tf::cpp::reindexed_by_ids_with_maps<Index, Real, Dims>(points, point_ids);
  CHECK((selected_points.points.raw_shape() ==
         tf::small_vector<int, 3>{2, static_cast<int>(Dims)}));
  CHECK(has_selected_points(selected_points.points, points, {4, 1}));
  CHECK(has_values(selected_points.point_map.kept_ids, {Index{4}, Index{1}}));
  CHECK(
      has_values(selected_points.point_map.f,
                 {Index{6}, Index{1}, Index{6}, Index{6}, Index{0}, Index{6}}));

  const auto raw_mask = make_array<std::int8_t>({0, 1, 0, 0, 1, 0}, {6});
  const auto masked_points =
      tf::cpp::reindexed_by_mask_with_maps<Index, Real, Dims>(points, raw_mask);
  CHECK(has_selected_points(masked_points.points, points, {1, 4}));
  CHECK(has_values(masked_points.point_map.kept_ids, {Index{1}, Index{4}}));

  const auto face_ids = make_array<Index>({Index{2}, Index{0}}, {2});
  const auto face_mask = make_array<std::int8_t>({1, 0, 1, 0}, {4});
  const auto kept_point_ids = make_array<Index>(
      {Index{5}, Index{4}, Index{3}, Index{5}, Index{4}}, {5});
  const auto kept_point_mask = make_array<std::int8_t>({0, 0, 0, 1, 1, 1}, {6});
  check_reindex_selector_case<TestType, 3>(face_ids, face_mask, kept_point_ids,
                                           kept_point_mask);
  check_reindex_selector_case<TestType, tf::dynamic_size>(
      face_ids, face_mask, kept_point_ids, kept_point_mask);

  auto tuple_fixed = fixed_python_mesh<TestType>();
  const auto fixed_faces = tf::cpp::test::face_indices_of(tuple_fixed.polygons);
  const auto fixed_points = points_array(tuple_fixed.polygons.points_buffer());
  using index_array = tf::cpp::nd_array<Index>;
  using point_array = tf::cpp::nd_array<Real>;
  using mask_array = tf::cpp::nd_array<std::int8_t>;
  using mesh_result = tf::cpp::reindexed_mesh_result<Index, Real, Dims>;
  using mesh_carrier = typename TestType::mesh_result_type;
  auto (*v3_ids_symbol)(const index_array &, const point_array &,
                        const index_array &)
      ->mesh_carrier = &tf::cpp::reindexed_by_ids<Index, Real, Dims, 3>;
  auto (*v3_ids_maps_symbol)(const index_array &, const point_array &,
                             const index_array &)
      ->mesh_result =
      &tf::cpp::reindexed_by_ids_with_maps<Index, Real, Dims, 3>;
  auto (*v3_mask_symbol)(const index_array &, const point_array &,
                         const mask_array &)
      ->mesh_carrier = &tf::cpp::reindexed_by_mask<Index, Real, Dims, 3>;
  auto (*v3_mask_maps_symbol)(const index_array &, const point_array &,
                              const mask_array &)
      ->mesh_result =
      &tf::cpp::reindexed_by_mask_with_maps<Index, Real, Dims, 3>;
  auto (*v3_point_ids_symbol)(const index_array &, const point_array &,
                              const index_array &)
      ->mesh_carrier =
      &tf::cpp::reindexed_by_ids_on_points<Index, Real, Dims, 3>;
  auto (*v3_point_mask_symbol)(const index_array &, const point_array &,
                               const mask_array &)
      ->mesh_carrier =
      &tf::cpp::reindexed_by_mask_on_points<Index, Real, Dims, 3>;

  CHECK(v3_ids_symbol(fixed_faces, fixed_points, face_ids).size() == 2);
  CHECK(v3_ids_maps_symbol(fixed_faces, fixed_points, face_ids).mesh.size() ==
        2);
  CHECK(v3_mask_symbol(fixed_faces, fixed_points, face_mask).size() == 2);
  CHECK(v3_mask_maps_symbol(fixed_faces, fixed_points, face_mask).mesh.size() ==
        2);
  CHECK(v3_point_ids_symbol(fixed_faces, fixed_points, kept_point_ids).size() ==
        2);
  CHECK(
      v3_point_mask_symbol(fixed_faces, fixed_points, kept_point_mask).size() ==
      2);

  CHECK(tf::cpp::async::reindexed_by_ids<Index, Real, Dims, 3>(
            fixed_faces, fixed_points, face_ids)
            .get()
            .size() == 2);
  CHECK(tf::cpp::async::reindexed_by_ids_with_maps<Index, Real, Dims, 3>(
            fixed_faces, fixed_points, face_ids)
            .get()
            .mesh.size() == 2);
  CHECK(tf::cpp::async::reindexed_by_mask<Index, Real, Dims, 3>(
            fixed_faces, fixed_points, face_mask)
            .get()
            .size() == 2);
  CHECK(tf::cpp::async::reindexed_by_mask_with_maps<Index, Real, Dims, 3>(
            fixed_faces, fixed_points, face_mask)
            .get()
            .mesh.size() == 2);
  CHECK(tf::cpp::async::reindexed_by_ids_on_points<Index, Real, Dims, 3>(
            fixed_faces, fixed_points, kept_point_ids)
            .get()
            .size() == 2);
  CHECK(tf::cpp::async::reindexed_by_mask_on_points<Index, Real, Dims, 3>(
            fixed_faces, fixed_points, kept_point_mask)
            .get()
            .size() == 2);

  const auto tuple_ids = tf::cpp::reindexed_by_ids_with_maps<Index, Real, Dims>(
      fixed_faces, fixed_points, face_ids);
  CHECK(tuple_ids.mesh.size() == 2);
  auto tuple_dynamic = dynamic_python_mesh<TestType>();
  const auto dynamic_faces =
      tf::cpp::offset_blocked_buffer<Index, Index>::create(
          tf::cpp::test::face_offsets_of(tuple_dynamic.polygons),
          tf::cpp::test::face_indices_of(tuple_dynamic.polygons));
  const auto dynamic_points =
      points_array(tuple_dynamic.polygons.points_buffer());
  const auto tuple_mask =
      tf::cpp::reindexed_by_mask_with_maps<Index, Real, Dims>(
          dynamic_faces, dynamic_points, face_mask);
  CHECK(tuple_mask.mesh.size() == 2);
  const auto tuple_point_mask =
      tf::cpp::reindexed_by_mask_on_points<Index, Real, Dims>(
          dynamic_faces, dynamic_points, kept_point_mask);
  CHECK(tuple_point_mask.size() == 2);

  auto edges = python_edge_mesh<TestType>();
  const auto original_edges = edges_array(edges.segments);
  const auto edge_points = points_array(edges.segments.points_buffer());
  const auto edge_ids = make_array<Index>({Index{2}, Index{0}}, {2});
  const auto edge_mask = make_array<std::int8_t>({0, 1, 0}, {3});
  const auto edge_by_ids =
      tf::cpp::reindexed_by_ids_with_maps<Index, Real, Dims>(edges.edge_mesh(),
                                                             edge_ids);
  const auto edge_by_mask =
      tf::cpp::reindexed_by_mask_with_maps<Index, Real, Dims>(edges.edge_mesh(),
                                                              edge_mask);
  const auto edge_tuple =
      tf::cpp::reindexed_by_ids_with_maps<Index, Real, Dims, 2>(
          original_edges, edge_points, edge_ids);
  const auto edge_point_mask = make_array<std::int8_t>({1, 1, 1, 0, 0}, {5});
  const auto edge_point_tuple =
      tf::cpp::reindexed_by_mask_on_points<Index, Real, Dims, 2>(
          original_edges, edge_points, edge_point_mask);
  const auto edge_point_ids =
      make_array<Index>({Index{2}, Index{1}, Index{0}, Index{2}}, {4});
  const auto edge_by_point_ids =
      tf::cpp::reindexed_by_ids_on_points_with_maps<Index, Real, Dims>(
          edges.edge_mesh(), edge_point_ids);
  const auto edge_by_point_mask =
      tf::cpp::reindexed_by_mask_on_points_with_maps<Index, Real, Dims>(
          edges.edge_mesh(), edge_point_mask);
  CHECK(edge_by_ids.mesh.size() == 2);
  CHECK(edge_tuple.mesh.size() == 2);
  CHECK(edge_point_tuple.size() == 2);
  CHECK(same_map(edge_by_point_ids.edge_map, edge_by_point_mask.edge_map));
  CHECK(same_map(edge_by_point_ids.point_map, edge_by_point_mask.point_map));
  CHECK(same_array(edges_array(edge_by_point_ids.mesh),
                   edges_array(edge_by_point_mask.mesh)));
  CHECK(same_array(points_array(edge_by_point_ids.mesh.points_buffer()),
                   points_array(edge_by_point_mask.mesh.points_buffer())));
  CHECK(has_values(edge_by_ids.edge_map.kept_ids, {Index{2}, Index{0}}));
  CHECK(edge_by_mask.mesh.size() == 1);
  CHECK(has_values(edge_by_mask.edge_map.kept_ids, {Index{1}}));
  CHECK(same_array(edges_array(edges.segments), original_edges));
}

TEMPLATE_LIST_TEST_CASE("typed concatenate promotes dynamic layout and applies "
                        "mesh and edge frames",
                        "[cpp][reindex][concatenate][matrix][python-parity]",
                        matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  // a range is homogeneous, so it concatenates at its element's own arity
  auto first = dynamic_python_mesh<TestType>();
  auto second = dynamic_python_mesh<TestType>();
  second.place(translation<TestType>());
  const auto first_points = points_array(first.polygons.points_buffer());
  const auto second_points = points_array(second.polygons.points_buffer());
  const auto joined = tf::cpp::concatenate_meshes(
      std::vector<tf::cpp::mesh<Index, Real, Dims, tf::dynamic_size>>{
          first.mesh(), second.mesh()});
  CHECK(joined.size() == 8);
  CHECK(joined.points_buffer().size() == 12);
  CHECK(tf::cpp::test::face_offsets_of(joined).raw_shape() ==
        tf::small_vector<int, 3>{9});
  CHECK(tf::cpp::test::face_indices_of(joined)[12] == Index{6});
  CHECK(joined.points_buffer().data_buffer()[6 * Dims] == Real{100});
  CHECK(same_array(points_array(first.polygons.points_buffer()), first_points));
  CHECK(
      same_array(points_array(second.polygons.points_buffer()), second_points));

  auto fixed = fixed_python_mesh<TestType>();
  const auto singleton = tf::cpp::concatenate_meshes(
      std::vector<tf::cpp::mesh<Index, Real, Dims>>{fixed.mesh()});
  static_assert(std::is_same_v<std::decay_t<decltype(singleton)>,
                               typename TestType::mesh_result_type>);
  CHECK(same_array(tf::cpp::test::face_indices_of(singleton),
                   tf::cpp::test::face_indices_of(fixed.polygons)));

  auto first_edges = python_edge_mesh<TestType>();
  auto second_edges = python_edge_mesh<TestType>();
  second_edges.place(translation<TestType>());
  const auto joined_edges = tf::cpp::concatenate_edge_meshes<Index, Real, Dims>(
      {first_edges.edge_mesh(), second_edges.edge_mesh()});
  CHECK(joined_edges.size() == 6);
  CHECK(joined_edges.points_buffer().size() == 10);
  CHECK(joined_edges.edges_buffer().data_buffer()[6] == Index{5});
  CHECK(joined_edges.points_buffer().data_buffer()[5 * Dims] == Real{100});
}

TEST_CASE("heterogeneous pair concatenation promotes both dtypes layouts and "
          "composed frames in either direction",
          "[cpp][reindex][concatenate][heterogeneous][python-parity]") {
  using Low = matrix_row<std::int32_t, float, 2>;
  using High = matrix_row<std::int64_t, double, 2>;
  // one triangle carrier beside one mixed carrier can only be stated mixed
  using result_mesh =
      tf::polygons_buffer<std::int64_t, double, 2, tf::dynamic_size>;
  using result_edge_mesh = tf::segments_buffer<std::int64_t, double, 2>;

  auto fixed = fixed_python_mesh<Low>();
  auto dynamic = dynamic_python_mesh<High>();
  fixed.place({0.0F, -1.0F, 10.0F, 1.0F, 0.0F, 20.0F, 0.0F, 0.0F, 1.0F});
  dynamic.place({-1.0, 0.0, 30.0, 0.0, -1.0, 40.0, 0.0, 0.0, 1.0});

  const auto forward =
      tf::cpp::concatenate_meshes(fixed.mesh(), dynamic.mesh());
  const auto reverse =
      tf::cpp::concatenate_meshes(dynamic.mesh(), fixed.mesh());
  static_assert(std::is_same_v<std::decay_t<decltype(forward)>, result_mesh>);
  static_assert(std::is_same_v<std::decay_t<decltype(reverse)>, result_mesh>);
  CHECK(forward.size() == 8);
  CHECK(forward.points_buffer().size() == 12);
  CHECK(tf::cpp::test::face_indices_of(forward)[12] == std::int64_t{6});
  const auto &forward_points = forward.points_buffer().data_buffer();
  const auto &reverse_points = reverse.points_buffer().data_buffer();
  CHECK(forward_points[0] == 10.0);
  CHECK(forward_points[1] == 20.0);
  CHECK(forward_points[2] == 10.0);
  CHECK(forward_points[3] == 21.0);
  CHECK(forward_points[12] == 30.0);
  CHECK(forward_points[13] == 40.0);
  CHECK(forward_points[14] == 29.0);
  CHECK(forward_points[15] == 40.0);
  CHECK(reverse_points[0] == 30.0);
  CHECK(reverse_points[1] == 40.0);
  CHECK(reverse_points[12] == 10.0);
  CHECK(reverse_points[13] == 20.0);

  const auto fixed_faces = tf::cpp::test::face_indices_of(fixed.polygons);
  const auto fixed_points = points_array(fixed.polygons.points_buffer());
  const auto dynamic_faces =
      tf::cpp::offset_blocked_buffer<std::int64_t, std::int64_t>::create(
          tf::cpp::test::face_offsets_of(dynamic.polygons),
          tf::cpp::test::face_indices_of(dynamic.polygons));
  const auto dynamic_points = points_array(dynamic.polygons.points_buffer());
  const auto tuple_forward =
      tf::cpp::concatenate_meshes<std::int32_t, float, std::int64_t, double, 2,
                                  3, tf::dynamic_size>(
          fixed_faces, fixed_points, dynamic_faces, dynamic_points);
  const auto tuple_reverse =
      tf::cpp::concatenate_meshes<std::int64_t, double, std::int32_t, float, 2,
                                  tf::dynamic_size, 3>(
          dynamic_faces, dynamic_points, fixed_faces, fixed_points);
  static_assert(
      std::is_same_v<std::decay_t<decltype(tuple_forward)>, result_mesh>);
  static_assert(
      std::is_same_v<std::decay_t<decltype(tuple_reverse)>, result_mesh>);
  CHECK(tuple_forward.points_buffer().data_buffer()[0] == 0.0);
  CHECK(tuple_forward.points_buffer().data_buffer()[12] == 0.0);

  auto fixed_edges = python_edge_mesh<Low>();
  auto high_edges = python_edge_mesh<High>();
  fixed_edges.place({0.0F, -1.0F, 10.0F, 1.0F, 0.0F, 20.0F, 0.0F, 0.0F, 1.0F});
  high_edges.place({-1.0, 0.0, 30.0, 0.0, -1.0, 40.0, 0.0, 0.0, 1.0});
  const auto forward_edges = tf::cpp::concatenate_edge_meshes(
      fixed_edges.edge_mesh(), high_edges.edge_mesh());
  const auto reverse_edges = tf::cpp::concatenate_edge_meshes(
      high_edges.edge_mesh(), fixed_edges.edge_mesh());
  static_assert(
      std::is_same_v<std::decay_t<decltype(forward_edges)>, result_edge_mesh>);
  static_assert(
      std::is_same_v<std::decay_t<decltype(reverse_edges)>, result_edge_mesh>);
  CHECK(forward_edges.size() == 6);
  CHECK(forward_edges.edges_buffer().data_buffer()[6] == std::int64_t{5});
  CHECK(forward_edges.points_buffer().data_buffer()[0] == 10.0);
  CHECK(forward_edges.points_buffer().data_buffer()[1] == 20.0);
  CHECK(forward_edges.points_buffer().data_buffer()[10] == 30.0);
  CHECK(forward_edges.points_buffer().data_buffer()[11] == 40.0);
  CHECK(reverse_edges.points_buffer().data_buffer()[0] == 30.0);
  CHECK(reverse_edges.points_buffer().data_buffer()[1] == 40.0);

  const auto tuple_edges =
      tf::cpp::concatenate_edge_meshes<std::int32_t, float, std::int64_t,
                                       double, 2>(
          edges_array(fixed_edges.segments),
          points_array(fixed_edges.segments.points_buffer()),
          edges_array(high_edges.segments),
          points_array(high_edges.segments.points_buffer()));
  static_assert(
      std::is_same_v<std::decay_t<decltype(tuple_edges)>, result_edge_mesh>);
  CHECK(tuple_edges.size() == 6);
  CHECK(tuple_edges.points_buffer().data_buffer()[0] == 0.0);
  CHECK(tuple_edges.points_buffer().data_buffer()[10] == 0.0);

  auto pending =
      tf::cpp::async::concatenate_meshes(fixed.mesh(), dynamic.mesh());
  static_assert(std::is_same_v<decltype(pending), std::future<result_mesh>>);
  const auto async_result = pending.get();
  CHECK(async_result.points_buffer().data_buffer()[0] == 10.0);
  CHECK(async_result.points_buffer().data_buffer()[12] == 30.0);
}

TEMPLATE_LIST_TEST_CASE("typed domain split preserves index dtype layout "
                        "empties and async snapshots",
                        "[cpp][reindex][split-domains][matrix][python-parity]",
                        matrix_rows) {
  using Index = typename TestType::index_type;

  check_split_domains_case<TestType, 3>();
  check_split_domains_case<TestType, tf::dynamic_size>();

  tf::cpp::test::mixed_mesh_of<typename TestType::mesh_type> empty_mesh;
  const auto empty = tf::cpp::split_into_domains(
      empty_mesh.mesh(), tf::cpp::domain_labels_result<Index>{});
  CHECK(empty.components.empty());
  CHECK(empty.labels.raw_shape() == tf::small_vector<int, 3>{0});
}

TEMPLATE_LIST_TEST_CASE(
    "typed reindex and concatenate materialize canonical empty carriers",
    "[cpp][reindex][typed][empty]", matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;

  const auto points = python_points<TestType>();
  const auto no_ids = make_array<Index>({}, {0});
  const auto no_points =
      tf::cpp::reindexed_by_ids_with_maps<Index, Real, Dims>(points, no_ids);
  CHECK((no_points.points.raw_shape() ==
         tf::small_vector<int, 3>{0, static_cast<int>(Dims)}));
  CHECK(no_points.point_map.f.raw_shape() == tf::small_vector<int, 3>{6});
  CHECK(no_points.point_map.kept_ids.raw_shape() ==
        tf::small_vector<int, 3>{0});

  const auto no_faces = make_array<std::int8_t>({0, 0, 0, 0}, {4});
  auto fixed = fixed_python_mesh<TestType>();
  auto dynamic = dynamic_python_mesh<TestType>();
  tf::cpp::test::owned_mesh<Index, Real, Dims> empty_fixed{
      tf::cpp::reindexed_by_mask<Index, Real, Dims>(fixed.mesh(), no_faces)};
  tf::cpp::test::mixed_mesh_of<typename TestType::mesh_type> empty_dynamic{
      tf::cpp::reindexed_by_mask<Index, Real, Dims>(dynamic.mesh(), no_faces)};
  CHECK(empty_fixed.polygons.size() == 0);
  CHECK(empty_fixed.polygons.points_buffer().size() == 0);
  CHECK(has_values(tf::cpp::test::face_offsets_of(empty_dynamic.polygons),
                   {Index{0}}));
  CHECK(tf::cpp::test::face_indices_of(empty_dynamic.polygons).raw_shape() ==
        tf::small_vector<int, 3>{0});

  const auto joined =
      tf::cpp::concatenate_meshes(empty_fixed.mesh(), empty_dynamic.mesh());
  static_assert(std::is_same_v<std::decay_t<decltype(joined)>,
                               typename TestType::dynamic_mesh_result_type>);
  CHECK(joined.size() == 0);
  CHECK(joined.points_buffer().size() == 0);
  CHECK(has_values(tf::cpp::test::face_offsets_of(joined), {Index{0}}));

  auto edges = python_edge_mesh<TestType>();
  tf::cpp::test::owned_edge_mesh<Index, Real, Dims> empty_edges{
      tf::cpp::reindexed_by_mask<Index, Real, Dims>(
          edges.edge_mesh(), make_array<std::int8_t>({0, 0, 0}, {3}))};
  CHECK(empty_edges.segments.size() == 0);
  CHECK(empty_edges.segments.points_buffer().size() == 0);
  const auto joined_edges = tf::cpp::concatenate_edge_meshes<Index, Real, Dims>(
      {empty_edges.edge_mesh()});
  CHECK(joined_edges.size() == 0);
}

/// An async selection carries its arrays as the handles they are and reads
/// the caller's own carrier, so a caller may release its handles before the
/// job runs and the storage the job reads stays.
TEMPLATE_LIST_TEST_CASE("typed selector and concatenate async APIs retain "
                        "their arrays past the caller's handle",
                        "[cpp][reindex][typed][async][ownership]",
                        matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;
  constexpr auto Dims = TestType::dims;
  auto submissions = std::make_shared<std::atomic<int>>(0);

  auto points = python_points<TestType>();
  auto point_ids = make_array<Index>({Index{4}, Index{1}}, {2});
  auto raw_pending = tf::cpp::async::reindexed_by_ids<Index, Real, Dims>(
      mutating_resolver{submissions,
                        [&] {
                          points.destroy();
                          point_ids.destroy();
                        }},
      points, point_ids);
  CHECK(has_selected_points(raw_pending.get(), python_points<TestType>(),
                            {4, 1}));

  auto tuple_source = dynamic_python_mesh<TestType>();
  auto tuple_faces = tf::cpp::offset_blocked_buffer<Index, Index>::create(
      tf::cpp::test::face_offsets_of(tuple_source.polygons),
      tf::cpp::test::face_indices_of(tuple_source.polygons));
  auto tuple_points = points_array(tuple_source.polygons.points_buffer());
  auto tuple_mask = make_array<std::int8_t>({1, 0, 1, 0}, {4});
  auto tuple_pending =
      tf::cpp::async::reindexed_by_mask_with_maps<Index, Real, Dims>(
          mutating_resolver{submissions,
                            [&] {
                              tuple_faces = {};
                              tuple_points.destroy();
                              tuple_mask.destroy();
                            }},
          tuple_faces, tuple_points, tuple_mask);
  const auto tuple_result = tuple_pending.get();
  CHECK(tuple_result.mesh.size() == 2);
  CHECK(tuple_result.mesh.points_buffer().data_buffer()[0] == Real{0});

  auto tuple_edges = python_edge_mesh<TestType>();
  auto tuple_edge_data = edges_array(tuple_edges.segments);
  auto tuple_edge_points = points_array(tuple_edges.segments.points_buffer());
  auto tuple_edge_ids = make_array<Index>({Index{2}, Index{0}}, {2});
  auto tuple_edge_pending =
      tf::cpp::async::reindexed_by_ids_with_maps<Index, Real, Dims, 2>(
          mutating_resolver{submissions,
                            [&] {
                              tuple_edge_data.destroy();
                              tuple_edge_points.destroy();
                              tuple_edge_ids.destroy();
                            }},
          tuple_edge_data, tuple_edge_points, tuple_edge_ids);
  const auto tuple_edge_result = tuple_edge_pending.get();
  CHECK(tuple_edge_result.mesh.size() == 2);
  CHECK(tuple_edge_result.mesh.points_buffer().data_buffer()[0] != Real{77});

  auto mesh = dynamic_python_mesh<TestType>();
  auto face_ids = make_array<Index>({Index{2}, Index{0}}, {2});
  auto mesh_pending =
      tf::cpp::async::reindexed_by_ids<mutating_resolver, Index, Real, Dims,
                                       tf::dynamic_size>(
          mutating_resolver{submissions, [&] { face_ids.destroy(); }},
          mesh.mesh(), face_ids);
  CHECK(mesh_pending.get().size() == 2);

  auto first = dynamic_python_mesh<TestType>();
  auto second = dynamic_python_mesh<TestType>();
  second.place(translation<TestType>());
  std::vector<tf::cpp::mesh<Index, Real, Dims, tf::dynamic_size>> meshes{
      first.mesh(), second.mesh()};
  auto concatenate_pending =
      tf::cpp::async::concatenate_meshes<mutating_resolver, Index, Real, Dims,
                                         tf::dynamic_size>(
          mutating_resolver{submissions, [&] { meshes.clear(); }}, meshes);
  const auto joined = concatenate_pending.get();
  CHECK(joined.size() == 8);
  CHECK(joined.points_buffer().data_buffer()[6 * Dims] == Real{100});

  auto first_edges = python_edge_mesh<TestType>();
  auto second_edges = python_edge_mesh<TestType>();
  std::vector<tf::cpp::edge_mesh<Index, Real, Dims>> edge_meshes{
      first_edges.edge_mesh(), second_edges.edge_mesh()};
  auto edge_pending =
      tf::cpp::async::concatenate_edge_meshes<mutating_resolver, Index, Real,
                                              Dims>(
          mutating_resolver{submissions, [&] { edge_meshes.clear(); }},
          edge_meshes);
  CHECK(edge_pending.get().size() == 6);
  CHECK(submissions->load(std::memory_order_relaxed) == 6);
}

TEST_CASE("typed reindex validates rank size values and indices before kernels",
          "[cpp][reindex][typed][validation]") {
  using Real = double;
  using Index = std::int64_t;
  constexpr std::size_t Dims = 2;
  using Row = matrix_row<Index, Real, Dims>;
  const auto points = python_points<Row>();

  CHECK_THROWS_AS((tf::cpp::reindexed_by_ids<Index, Real, Dims>(
                      points, make_array<Index>({Index{0}}, {1, 1}))),
                  std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::reindexed_by_ids<Index, Real, Dims>(
                      points, make_array<Index>({Index{-1}}, {1}))),
                  std::out_of_range);
  CHECK_THROWS_AS((tf::cpp::reindexed_by_mask<Index, Real, Dims>(
                      points, make_array<std::int8_t>({1}, {1}))),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      (tf::cpp::reindexed_by_mask<Index, Real, Dims>(
          points, make_array<std::int8_t>({1, 0, 0, 0, 0, 2}, {6}))),
      std::invalid_argument);

  // the cache owns the indices for the reading it answers, so a read refuses
  // every corner that names a point the geometry does not have
  auto invalid = fixed_python_mesh<Row>();
  invalid.polygons.faces_buffer().data_buffer()[0] = Index{6};
  invalid.cache.faces_changed();
  CHECK_THROWS_AS(
      (tf::cpp::reindexed_by_mask<Index, Real, Dims>(
          invalid.mesh(), make_array<std::int8_t>({1, 0, 0, 0}, {4}))),
      std::out_of_range);

  auto valid_for_point_ids = fixed_python_mesh<Row>();
  CHECK_THROWS_AS((tf::cpp::reindexed_by_ids_on_points<Index, Real, Dims>(
                      valid_for_point_ids.mesh(),
                      make_array<Index>({Index{-1}, Index{0}}, {2}))),
                  std::out_of_range);
  CHECK_THROWS_AS(
      (tf::cpp::reindexed_by_ids_on_points<Index, Real, Dims>(
          valid_for_point_ids.mesh(), make_array<Index>({Index{6}}, {1}))),
      std::out_of_range);
  auto valid_edges_for_point_ids = python_edge_mesh<Row>();
  CHECK_THROWS_AS((tf::cpp::reindexed_by_ids_on_points<Index, Real, Dims>(
                      valid_edges_for_point_ids.edge_mesh(),
                      make_array<Index>({Index{5}}, {1}))),
                  std::out_of_range);

  const auto domain_max = std::numeric_limits<Index>::max();
  const auto oversized_domains = tf::cpp::domain_labels_result<Index>(
      make_array<Index>({domain_max, domain_max, domain_max, domain_max,
                         domain_max, domain_max, domain_max, domain_max},
                        {4, 2}),
      domain_max, Index{-1});
  auto valid_for_domains = fixed_python_mesh<Row>();
  CHECK_THROWS_AS((tf::cpp::split_into_domains<Index, Real, Dims>(
                      valid_for_domains.mesh(), oversized_domains)),
                  std::invalid_argument);

  CHECK_THROWS_AS(tf::cpp::concatenate_meshes(
                      std::vector<tf::cpp::mesh<Index, Real, Dims>>{}),
                  std::invalid_argument);
  CHECK_THROWS_AS((tf::cpp::concatenate_edge_meshes<Index, Real, Dims>({})),
                  std::invalid_argument);
}

TEST_CASE("the split-components signature and result are one",
          "[cpp][reindex][split-components][abi]") {
  using result_type = tf::cpp::split_components_result<
      tf::polygons_buffer<tf::cpp::default_index_t, float, 3, 3>>;
  using function_type =
      result_type (*)(const tf::cpp::mesh<tf::cpp::default_index_t, float> &,
                      const tf::cpp::nd_array<std::int32_t> &);
  const function_type symbol =
      &tf::cpp::split_into_components<std::int32_t, float, 3>;
  REQUIRE(symbol != nullptr);

  auto mesh = fixed_python_mesh<matrix_row<std::int32_t, float, 3>>();
  auto result = symbol(
      mesh.mesh(), make_array<std::int32_t>({std::int32_t{7}, std::int32_t{7},
                                             std::int32_t{3}, std::int32_t{3}},
                                            {4}));
  static_assert(std::is_same_v<decltype(result), result_type>);
  REQUIRE(result.components.size() == 2);
  CHECK(has_values(result.labels, {std::int32_t{3}, std::int32_t{7}}));
}

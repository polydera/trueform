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

#include "trueform/cpp/clean.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <future>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

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

template <typename Real>
using owned_t = tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real>;

template <typename Real>
using cleaned_t = tf::polygons_buffer<tf::cpp::default_index_t, Real, 3, 3>;

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

template <typename Real> auto duplicate_mesh() -> owned_t<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 3, 1, 2}, {0, 0, 0, 1, 0, 0, Real{0.5}, 1, 0, 0, 0, 0})};
}

template <typename Real> auto duplicate_points() -> tf::cpp::nd_array<Real> {
  return make_array<Real>({0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0}, {4, 3});
}

template <typename Real> auto zero_face_mesh() -> owned_t<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {}, {0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0})};
}

template <typename Real>
auto near_duplicate_points() -> tf::cpp::nd_array<Real> {
  return make_array<Real>({0, 0, 0, Real{0.009}, 0, 0, 1, 1, 0}, {3, 3});
}

template <typename Real> auto near_degenerate_mesh() -> owned_t<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2}, {0, 0, 0, Real{0.009}, 0, 0, 1, 1, 0})};
}

template <typename Real>
auto near_degenerate_soup() -> tf::cpp::nd_array<Real> {
  auto points = near_duplicate_points<Real>();
  points.set_shape({1, 3, 3});
  return points;
}

template <typename Real> auto triangle_soup() -> tf::cpp::nd_array<Real> {
  return make_array<Real>(
      {0, 0, 0, 1, 0, 0, Real{0.5}, 1, 0, 1, 0, 0, 0, 0, 0, Real{0.5}, -1, 0},
      {2, 3, 3});
}

template <typename Real> auto affine_transform() -> std::array<Real, 16> {
  return {1000, 0, 0, 7, 0, 1000, 0, 8, 0, 0, 1000, 9, 0, 0, 0, 1};
}

template <typename T>
auto same_values(const tf::cpp::nd_array<T> &a, const tf::cpp::nd_array<T> &b)
    -> bool {
  if (a.raw_shape() != b.raw_shape())
    return false;
  for (std::size_t index = 0; index < a.length(); ++index)
    if (a[index] != b[index])
      return false;
  return true;
}

template <typename T>
auto same_storage(const tf::buffer<T> &a, const tf::buffer<T> &b) -> bool {
  if (a.size() != b.size())
    return false;
  for (std::size_t index = 0; index < a.size(); ++index)
    if (a[index] != b[index])
      return false;
  return true;
}

template <typename Real>
auto check_duplicate_mesh(const cleaned_t<Real> &value) -> void {
  CHECK(value.size() == 1);
  CHECK(value.points_buffer().size() == 3);
  CHECK(value.faces_buffer().data_buffer().size() == 3);
  CHECK(value.points_buffer().data_buffer().size() == 9);
}

} // namespace

TEMPLATE_TEST_CASE("clean facade preserves every typed operation and maps",
                   "[cpp][clean][sync]", float, double) {
  auto mesh = duplicate_mesh<TestType>();
  auto points = duplicate_points<TestType>();
  auto soup = triangle_soup<TestType>();

  const auto cleaned_mesh = tf::cpp::cleaned_mesh(mesh.mesh());
  const auto mesh_maps = tf::cpp::cleaned_mesh_with_maps(mesh.mesh());
  const auto cleaned_points = tf::cpp::cleaned_points(points);
  const auto point_maps = tf::cpp::cleaned_points_with_map(points);
  const auto cleaned_soup = tf::cpp::cleaned_polygon_soup(soup);

  static_assert(
      std::is_same_v<decltype(cleaned_mesh), const cleaned_t<TestType>>);
  static_assert(std::is_same_v<decltype(cleaned_points),
                               const tf::cpp::nd_array<TestType>>);
  check_duplicate_mesh<TestType>(cleaned_mesh);
  check_duplicate_mesh<TestType>(mesh_maps.mesh);
  CHECK(mesh_maps.face_map.is_valid());
  CHECK(mesh_maps.point_map.is_valid());
  CHECK(mesh_maps.face_map.f.length() == 2);
  CHECK(mesh_maps.face_map.kept_ids.length() == 1);
  CHECK(mesh_maps.point_map.f.length() == 4);
  CHECK(mesh_maps.point_map.kept_ids.length() == 3);

  CHECK((cleaned_points.raw_shape() == tf::small_vector<int, 3>{3, 3}));
  CHECK((point_maps.points.raw_shape() == tf::small_vector<int, 3>{3, 3}));
  CHECK(point_maps.point_map.f.length() == 4);
  CHECK(point_maps.point_map.kept_ids.length() == 3);
  CHECK(cleaned_soup.size() == 2);
  CHECK(cleaned_soup.points_buffer().size() == 4);

  CHECK(mesh.polygons.size() == 2);
  CHECK(mesh.polygons.points_buffer().size() == 4);
  CHECK(points.length() == 12);
  CHECK(soup.length() == 18);
}

TEMPLATE_TEST_CASE("clean facade preserves tolerance and mesh options",
                   "[cpp][clean][options]", float, double) {
  const auto delta = TestType{0.009};
  auto points = make_array<TestType>({0, 0, 0, delta, 0, 0, 1, 0, 0}, {3, 3});
  CHECK(tf::cpp::cleaned_points(points).shape_at(0) == 3);
  CHECK(tf::cpp::cleaned_points(points, TestType{0.02}).shape_at(0) == 2);
  CHECK_THROWS_AS(tf::cpp::cleaned_points(points, TestType{-1}),
                  std::invalid_argument);

  owned_t<TestType> duplicate_faces{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, 2, 0, 1, 2}, {0, 0, 0, 1, 0, 0, 0, 1, 0})};
  CHECK(tf::cpp::cleaned_mesh(duplicate_faces.mesh()).size() == 1);
  CHECK(
      tf::cpp::cleaned_mesh(duplicate_faces.mesh(), TestType{}, false).size() ==
      2);

  owned_t<TestType> unreferenced{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, 2}, {0, 0, 0, 1, 0, 0, 0, 1, 0, 9, 9, 9})};
  CHECK(tf::cpp::cleaned_mesh(unreferenced.mesh()).points_buffer().size() == 3);
  CHECK(tf::cpp::cleaned_mesh(unreferenced.mesh(), TestType{}, true, false)
            .points_buffer()
            .size() == 4);

  const auto ignored = tf::cpp::cleaned_polygon_soup(triangle_soup<TestType>(),
                                                     TestType{}, false, false);
  CHECK(ignored.size() == 2);
  const auto ignored_point_options = tf::cpp::cleaned_points(
      duplicate_points<TestType>(), TestType{}, false, false);
  CHECK(ignored_point_options.shape_at(0) == 3);
}

TEMPLATE_TEST_CASE("clean facade preserves local transformed mesh semantics",
                   "[cpp][clean][transform]", float, double) {
  owned_t<TestType> input{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 2, 3, 1, 2, 3},
          {0, 0, 0, TestType{0.009}, 0, 0, 1, 0, 0, 0, 1, 0})};
  input.place(affine_transform<TestType>());

  const auto result = tf::cpp::cleaned_mesh(input.mesh(), TestType{0.02});
  const auto mapped =
      tf::cpp::cleaned_mesh_with_maps(input.mesh(), TestType{0.02});

  CHECK(result.points_buffer().size() == 3);
  CHECK(mapped.mesh.points_buffer().size() == 3);
  CHECK(result.points_buffer().data_buffer()[0] == TestType{0});
  CHECK(result.points_buffer().data_buffer()[1] == TestType{0});
  CHECK(result.points_buffer().data_buffer()[2] == TestType{0});
}

TEMPLATE_TEST_CASE("zero-face mesh cleaning preserves point and map contracts",
                   "[cpp][clean][zero-face]", float, double) {
  auto keep_input = zero_face_mesh<TestType>();
  auto remove_input = zero_face_mesh<TestType>();

  const auto kept =
      tf::cpp::cleaned_mesh(keep_input.mesh(), TestType{}, true, false);
  const auto kept_maps = tf::cpp::cleaned_mesh_with_maps(
      keep_input.mesh(), TestType{}, true, false);
  const auto removed =
      tf::cpp::cleaned_mesh(remove_input.mesh(), TestType{}, true, true);
  const auto removed_maps = tf::cpp::cleaned_mesh_with_maps(
      remove_input.mesh(), TestType{}, true, true);

  CHECK(kept.size() == 0);
  CHECK(kept.points_buffer().size() == 3);
  CHECK(kept_maps.mesh.size() == 0);
  CHECK(kept_maps.mesh.points_buffer().size() == 3);
  CHECK(kept_maps.face_map.f.length() == 0);
  CHECK(kept_maps.face_map.kept_ids.length() == 0);
  CHECK(kept_maps.point_map.f.length() == 4);
  CHECK(kept_maps.point_map.kept_ids.length() == 3);
  CHECK(kept_maps.point_map.f[0] == 0);
  CHECK(kept_maps.point_map.f[1] == 1);
  CHECK(kept_maps.point_map.f[2] == 0);
  CHECK(kept_maps.point_map.f[3] == 2);

  owned_t<TestType> tolerance_input{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {}, {0, 0, 0, TestType{0.009}, 0, 0, 1, 1, 0})};
  const auto tolerance_maps = tf::cpp::cleaned_mesh_with_maps(
      tolerance_input.mesh(), TestType{0.02}, true, false);
  CHECK(tolerance_maps.mesh.points_buffer().size() == 2);
  CHECK(tolerance_maps.point_map.f.length() == 3);
  CHECK(tolerance_maps.point_map.kept_ids.length() == 2);

  CHECK(removed.size() == 0);
  CHECK(removed.points_buffer().size() == 0);
  CHECK(removed_maps.mesh.size() == 0);
  CHECK(removed_maps.mesh.points_buffer().size() == 0);
  CHECK(removed_maps.point_map.f.length() == 4);
  CHECK(removed_maps.point_map.kept_ids.length() == 0);
  for (const auto mapped : removed_maps.point_map.f)
    CHECK(mapped == 4);

  auto async_kept =
      tf::cpp::async::cleaned_mesh(keep_input.mesh(), TestType{}, true, false);
  auto async_kept_maps = tf::cpp::async::cleaned_mesh_with_maps(
      keep_input.mesh(), TestType{}, true, false);
  auto async_removed =
      tf::cpp::async::cleaned_mesh(remove_input.mesh(), TestType{}, true, true);
  auto async_removed_maps = tf::cpp::async::cleaned_mesh_with_maps(
      remove_input.mesh(), TestType{}, true, true);

  CHECK(async_kept.get().points_buffer().size() == 3);
  const auto kept_async_result = async_kept_maps.get();
  CHECK(kept_async_result.mesh.points_buffer().size() == 3);
  CHECK(kept_async_result.point_map.f.length() == 4);
  CHECK(kept_async_result.point_map.kept_ids.length() == 3);
  CHECK(async_removed.get().points_buffer().size() == 0);
  const auto removed_async_result = async_removed_maps.get();
  CHECK(removed_async_result.mesh.points_buffer().size() == 0);
  CHECK(removed_async_result.point_map.f.length() == 4);
  CHECK(removed_async_result.point_map.kept_ids.length() == 0);
}

TEMPLATE_TEST_CASE(
    "clean tolerances share exact and finite validation semantics",
    "[cpp][clean][tolerance]", float, double) {
  const auto exact_tolerances =
      std::array<TestType, 2>{TestType{0}, -TestType{0}};
  for (const auto tolerance : exact_tolerances) {
    auto points = near_duplicate_points<TestType>();
    auto soup = near_degenerate_soup<TestType>();
    auto mesh = near_degenerate_mesh<TestType>();
    CHECK(tf::cpp::cleaned_points(points, tolerance).shape_at(0) == 3);
    CHECK(tf::cpp::cleaned_points_with_map(points, tolerance)
              .point_map.kept_ids.length() == 3);
    CHECK(tf::cpp::cleaned_polygon_soup(soup, tolerance).size() == 1);
    CHECK(tf::cpp::cleaned_mesh(mesh.mesh(), tolerance).size() == 1);
    CHECK(tf::cpp::cleaned_mesh_with_maps(mesh.mesh(), tolerance).mesh.size() ==
          1);
  }

  const auto nonfinite_tolerances =
      std::array<TestType, 3>{std::numeric_limits<TestType>::quiet_NaN(),
                              std::numeric_limits<TestType>::infinity(),
                              -std::numeric_limits<TestType>::infinity()};
  for (const auto tolerance : nonfinite_tolerances) {
    auto points = duplicate_points<TestType>();
    auto soup = triangle_soup<TestType>();
    auto mesh = duplicate_mesh<TestType>();
    CHECK_THROWS_AS(tf::cpp::cleaned_points(points, tolerance),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::cleaned_points_with_map(points, tolerance),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::cleaned_polygon_soup(soup, tolerance),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::cleaned_mesh(mesh.mesh(), tolerance),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::cleaned_mesh_with_maps(mesh.mesh(), tolerance),
                    std::invalid_argument);

    CHECK_THROWS_AS(tf::cpp::async::cleaned_points(points, tolerance).get(),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        tf::cpp::async::cleaned_points_with_map(points, tolerance).get(),
        std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::async::cleaned_polygon_soup(soup, tolerance).get(),
                    std::invalid_argument);
    CHECK_THROWS_AS(tf::cpp::async::cleaned_mesh(mesh.mesh(), tolerance).get(),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        tf::cpp::async::cleaned_mesh_with_maps(mesh.mesh(), tolerance).get(),
        std::invalid_argument);
  }
}

TEMPLATE_TEST_CASE("clean facade supports empty scalar and degenerate inputs",
                   "[cpp][clean][edge]", float, double) {
  auto empty_points = make_array<TestType>({}, {0, 3});
  auto empty_soup = make_array<TestType>({}, {0, 3, 3});
  owned_t<TestType> empty_mesh;
  auto scalar = make_array<TestType>({1, 2, 3}, {3});
  auto single_soup = make_array<TestType>({0, 0, 0, 1, 0, 0, 0, 1, 0}, {3, 3});
  auto degenerate =
      make_array<TestType>({0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 3, 3});

  CHECK(tf::cpp::cleaned_points(empty_points).raw_shape() ==
        tf::small_vector<int, 3>{0, 3});
  CHECK(tf::cpp::cleaned_points_with_map(empty_points).point_map.f.length() ==
        0);
  CHECK(tf::cpp::cleaned_polygon_soup(empty_soup).size() == 0);
  CHECK(tf::cpp::cleaned_mesh(empty_mesh.mesh()).size() == 0);
  CHECK(
      tf::cpp::cleaned_mesh_with_maps(empty_mesh.mesh()).face_map.f.length() ==
      0);
  CHECK(tf::cpp::cleaned_points(scalar).raw_shape() ==
        tf::small_vector<int, 3>{1, 3});
  CHECK(tf::cpp::cleaned_polygon_soup(single_soup).size() == 1);
  CHECK(tf::cpp::cleaned_polygon_soup(degenerate).size() == 0);
}

TEMPLATE_TEST_CASE("clean facade rejects malformed arrays shapes and indices",
                   "[cpp][clean][validation]", float, double) {
  tf::cpp::nd_array<TestType> invalid_array;
  owned_t<TestType> empty_mesh;
  CHECK_THROWS_AS(tf::cpp::cleaned_points(invalid_array),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::cleaned_points_with_map(invalid_array),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::cleaned_polygon_soup(invalid_array),
                  std::invalid_argument);
  // a mesh over empty storage is the EMPTY mesh, which cleans to itself
  CHECK(tf::cpp::cleaned_mesh(empty_mesh.mesh()).size() == 0);
  CHECK(tf::cpp::cleaned_mesh_with_maps(empty_mesh.mesh()).mesh.size() == 0);

  CHECK_THROWS_AS(tf::cpp::cleaned_points(
                      make_array<TestType>({0, 0, 0, 1, 1, 1}, {2, 3, 1})),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::cleaned_polygon_soup(
                      make_array<TestType>({0, 0, 0, 1, 1, 1}, {2, 3})),
                  std::invalid_argument);

  // the cache owns the indices for the reading it answers, so every corner
  // that names a point the geometry does not have is refused at the read
  owned_t<TestType> negative{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, -1}, {0, 0, 0, 1, 0, 0, 0, 1, 0})};
  CHECK_THROWS_AS(tf::cpp::cleaned_mesh(negative.mesh()), std::out_of_range);
  owned_t<TestType> beyond{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, 3}, {0, 0, 0, 1, 0, 0, 0, 1, 0})};
  CHECK_THROWS_AS(tf::cpp::cleaned_mesh(beyond.mesh()), std::out_of_range);

  owned_t<TestType> oversized{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, 2}, {0, 0, 0, 1, 0, 0, 0, 1, 0})};
  oversized.polygons.points_buffer().data_buffer().allocate(6);
  oversized.cache.points_changed();
  CHECK_THROWS_AS(tf::cpp::cleaned_mesh(oversized.mesh()), std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::cleaned_mesh_with_maps(oversized.mesh()),
                  std::out_of_range);
}

TEMPLATE_TEST_CASE("clean async retains its arrays and preserves exact types",
                   "[cpp][clean][async]", float, double) {
  auto soup = triangle_soup<TestType>();
  auto mesh = duplicate_mesh<TestType>();
  auto mapped_mesh = duplicate_mesh<TestType>();
  auto points = duplicate_points<TestType>();
  auto mapped_points = duplicate_points<TestType>();

  int submissions = 0;
  auto soup_result = tf::cpp::async::cleaned_polygon_soup(
      counting_resolver{&submissions}, soup);
  auto mesh_result = tf::cpp::async::cleaned_mesh(
      counting_resolver{&submissions}, mesh.mesh());
  auto mesh_maps_result = tf::cpp::async::cleaned_mesh_with_maps(
      counting_resolver{&submissions}, mapped_mesh.mesh());
  auto points_result =
      tf::cpp::async::cleaned_points(counting_resolver{&submissions}, points);
  auto point_maps_result = tf::cpp::async::cleaned_points_with_map(
      counting_resolver{&submissions}, mapped_points);

  static_assert(
      std::is_same_v<decltype(soup_result), std::future<cleaned_t<TestType>>>);
  static_assert(
      std::is_same_v<decltype(mesh_result), std::future<cleaned_t<TestType>>>);
  static_assert(std::is_same_v<decltype(mesh_maps_result),
                               std::future<tf::cpp::cleaned_mesh_result<
                                   tf::cpp::default_index_t, TestType>>>);
  static_assert(std::is_same_v<decltype(points_result),
                               std::future<tf::cpp::nd_array<TestType>>>);
  static_assert(std::is_same_v<decltype(point_maps_result),
                               std::future<tf::cpp::cleaned_points_result<
                                   tf::cpp::default_index_t, TestType>>>);
  CHECK(submissions == 5);

  soup.destroy();
  points.destroy();
  mapped_points.destroy();

  CHECK(soup_result.get().size() == 2);
  check_duplicate_mesh<TestType>(mesh_result.get());
  CHECK(mesh_maps_result.get().face_map.f.length() == 2);
  CHECK(points_result.get().shape_at(0) == 3);
  CHECK(point_maps_result.get().point_map.f.length() == 4);

  auto default_mesh = duplicate_mesh<TestType>();
  auto default_soup = triangle_soup<TestType>();
  auto default_points = duplicate_points<TestType>();
  CHECK(tf::cpp::async::cleaned_mesh(default_mesh.mesh()).get().size() == 1);
  CHECK(tf::cpp::async::cleaned_mesh_with_maps(default_mesh.mesh())
            .get()
            .face_map.f.length() == 2);
  CHECK(tf::cpp::async::cleaned_polygon_soup(default_soup).get().size() == 2);
  CHECK(tf::cpp::async::cleaned_points(default_points).get().shape_at(0) == 3);
  CHECK(tf::cpp::async::cleaned_points_with_map(default_points)
            .get()
            .point_map.f.length() == 4);

  tf::cpp::nd_array<TestType> invalid;
  auto array_failure = tf::cpp::async::cleaned_points(invalid);
  owned_t<TestType> empty_mesh;
  auto cleaned = tf::cpp::async::cleaned_mesh_with_maps(empty_mesh.mesh());
  CHECK_THROWS_AS(array_failure.get(), std::invalid_argument);
  CHECK(cleaned.get().mesh.size() == 0);
}

TEMPLATE_TEST_CASE("clean operations link from archive symbols",
                   "[cpp][clean][archive-link]", float, double) {
  auto (*soup_operation)(const tf::cpp::nd_array<TestType> &, TestType, bool,
                         bool)
      ->cleaned_t<TestType> =
      &tf::cpp::cleaned_polygon_soup<tf::cpp::default_index_t, TestType, 3, 3>;
  auto (*mesh_operation)(
      const tf::cpp::mesh<tf::cpp::default_index_t, TestType> &, TestType, bool,
      bool)
      ->cleaned_t<TestType> =
      &tf::cpp::cleaned_mesh<tf::cpp::default_index_t, TestType, 3, 3>;
  auto (*mesh_maps_operation)(
      const tf::cpp::mesh<tf::cpp::default_index_t, TestType> &, TestType, bool,
      bool)
      ->tf::cpp::cleaned_mesh_result<tf::cpp::default_index_t, TestType> =
      &tf::cpp::cleaned_mesh_with_maps<tf::cpp::default_index_t, TestType, 3,
                                       3>;
  auto (*points_operation)(const tf::cpp::nd_array<TestType> &, TestType, bool,
                           bool)
      ->tf::cpp::nd_array<TestType> = &tf::cpp::cleaned_points<TestType>;
  auto (*point_maps_operation)(const tf::cpp::nd_array<TestType> &, TestType,
                               bool, bool)
      ->tf::cpp::cleaned_points_result<tf::cpp::default_index_t, TestType> =
      &tf::cpp::cleaned_points_with_map<TestType>;

  auto mesh = duplicate_mesh<TestType>();
  auto points = duplicate_points<TestType>();
  CHECK(soup_operation(triangle_soup<TestType>(), TestType{}, true, true)
            .size() == 2);
  CHECK(mesh_operation(mesh.mesh(), TestType{}, true, true).size() == 1);
  CHECK(mesh_maps_operation(mesh.mesh(), TestType{}, true, true)
            .face_map.f.length() == 2);
  CHECK(points_operation(points, TestType{}, true, true).shape_at(0) == 3);
  CHECK(point_maps_operation(points, TestType{}, true, true)
            .point_map.f.length() == 4);
}

TEMPLATE_TEST_CASE("clean operations are deterministic across repeated runs",
                   "[cpp][clean][determinism]", float, double) {
  auto mesh = duplicate_mesh<TestType>();
  auto points = duplicate_points<TestType>();
  const auto expected_mesh = tf::cpp::cleaned_mesh_with_maps(mesh.mesh());
  const auto expected_points = tf::cpp::cleaned_points_with_map(points);

  for (int iteration = 0; iteration < 16; ++iteration) {
    const auto actual_mesh = tf::cpp::cleaned_mesh_with_maps(mesh.mesh());
    const auto actual_points = tf::cpp::cleaned_points_with_map(points);
    CHECK(same_storage(actual_mesh.mesh.faces_buffer().data_buffer(),
                       expected_mesh.mesh.faces_buffer().data_buffer()));
    CHECK(same_storage(actual_mesh.mesh.points_buffer().data_buffer(),
                       expected_mesh.mesh.points_buffer().data_buffer()));
    CHECK(same_values(actual_mesh.face_map.f, expected_mesh.face_map.f));
    CHECK(same_values(actual_mesh.point_map.f, expected_mesh.point_map.f));
    CHECK(same_values(actual_points.points, expected_points.points));
    CHECK(same_values(actual_points.point_map.f, expected_points.point_map.f));
  }
}

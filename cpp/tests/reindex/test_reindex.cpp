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
#include "trueform/cpp/reindex.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct counting_resolver {
  std::shared_ptr<std::atomic<int>> submissions;

  template <typename T>
  using state_type = tf::cpp::async::detail::future_state<T>;

  template <typename T>
  auto make_state() const -> std::shared_ptr<state_type<T>> {
    submissions->fetch_add(1, std::memory_order_relaxed);
    return std::make_shared<state_type<T>>();
  }
};

template <typename Real>
using owned_t = tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real>;

template <typename Real>
using selected_t = tf::polygons_buffer<tf::cpp::default_index_t, Real, 3, 3>;

template <typename T>
auto array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> buffer;
  buffer.allocate(values.size());
  auto output = buffer.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename Real> auto two_triangles() -> owned_t<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 3, 4, 5},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{1},
       Real{0}, Real{10}, Real{0}, Real{0}, Real{11}, Real{0}, Real{0},
       Real{10}, Real{1}, Real{0}})};
}

template <typename Real> auto duplicate_mesh() -> owned_t<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 2, 3, 1, 2, 3},
      {Real{0}, Real{0}, Real{0}, Real{0.009}, Real{0}, Real{0}, Real{1},
       Real{0}, Real{0}, Real{0}, Real{1}, Real{0}})};
}

template <typename Real> auto empty_mesh() -> owned_t<Real> { return {}; }

template <typename Real> auto translation(Real x) -> std::array<Real, 16> {
  return {Real{1}, Real{0}, Real{0}, x,       Real{0}, Real{1},
          Real{0}, Real{0}, Real{0}, Real{0}, Real{1}, Real{0},
          Real{0}, Real{0}, Real{0}, Real{1}};
}

template <typename Real> auto high_valence_fan() -> owned_t<Real> {
  constexpr int count = 256;
  owned_t<Real> owned;
  auto &points = owned.polygons.points_buffer().data_buffer();
  points.allocate(static_cast<std::size_t>((count + 1) * 3));
  points[0] = Real{0};
  points[1] = Real{0};
  points[2] = Real{0};
  for (int index = 0; index < count; ++index) {
    const auto angle = Real{6.2831853071795864769} * Real(index) / Real(count);
    const auto offset = static_cast<std::size_t>((index + 1) * 3);
    points[offset] = std::cos(angle);
    points[offset + 1] = std::sin(angle);
    points[offset + 2] = Real{0};
  }
  auto &faces = owned.polygons.faces_buffer().data_buffer();
  faces.allocate(static_cast<std::size_t>(count * 3));
  for (int index = 0; index < count; ++index) {
    const auto offset = static_cast<std::size_t>(index * 3);
    faces[offset] = 0;
    faces[offset + 1] = index + 1;
    faces[offset + 2] = (index + 1) % count + 1;
  }
  return owned;
}

template <typename T>
auto equal_array(const tf::cpp::nd_array<T> &first,
                 const tf::cpp::nd_array<T> &second) -> bool {
  return first.raw_shape() == second.raw_shape() &&
         first.length() == second.length() &&
         (first.empty() || std::memcmp(first.raw_data(), second.raw_data(),
                                       first.length() * sizeof(T)) == 0);
}

template <typename T>
auto equal_storage(const tf::buffer<T> &first, const tf::buffer<T> &second)
    -> bool {
  return first.size() == second.size() &&
         std::equal(first.begin(), first.end(), second.begin());
}

template <typename Real>
auto equal_mesh(const selected_t<Real> &first, const selected_t<Real> &second)
    -> bool {
  return equal_storage(first.faces_buffer().data_buffer(),
                       second.faces_buffer().data_buffer()) &&
         equal_storage(first.points_buffer().data_buffer(),
                       second.points_buffer().data_buffer());
}

} // namespace

TEMPLATE_TEST_CASE("reindex filters faces and points with typed owning maps",
                   "[cpp][reindex][sync]", float, double) {
  auto mesh = two_triangles<TestType>();
  const auto mask = array<std::int8_t>({1, 0}, {2});
  const auto ids = array<std::int32_t>({1}, {1});

  const auto by_mask = tf::cpp::reindexed_by_mask(mesh.mesh(), mask);
  const auto by_mask_maps =
      tf::cpp::reindexed_by_mask_with_maps(mesh.mesh(), mask);
  const auto by_ids = tf::cpp::reindexed_by_ids(mesh.mesh(), ids);
  const auto by_ids_maps =
      tf::cpp::reindexed_by_ids_with_maps(mesh.mesh(), ids);

  static_assert(std::is_same_v<decltype(by_mask), const selected_t<TestType>>);
  static_assert(std::is_same_v<decltype(by_mask_maps),
                               const tf::cpp::reindexed_mesh_result<
                                   tf::cpp::default_index_t, TestType>>);
  CHECK(by_mask.size() == 1);
  CHECK(by_mask.points_buffer().size() == 3);
  CHECK(by_mask_maps.face_map.f.raw_shape() == tf::small_vector<int, 3>{2});
  CHECK(by_mask_maps.face_map.kept_ids[0] == 0);
  CHECK(by_mask_maps.point_map.kept_ids.length() == 3);
  CHECK(by_ids.size() == 1);
  CHECK(by_ids.points_buffer().data_buffer()[0] == Catch::Approx(TestType{10}));
  CHECK(by_ids_maps.face_map.kept_ids[0] == 1);
  CHECK(by_ids_maps.point_map.kept_ids[0] == 3);

  mesh = {};
  CHECK(by_mask_maps.face_map.is_valid());
  CHECK(by_mask_maps.point_map.is_valid());
}

TEMPLATE_TEST_CASE("reindex point selectors preserve maps and empty behavior",
                   "[cpp][reindex][points]", float, double) {
  auto mesh = two_triangles<TestType>();
  const auto mask = array<std::int8_t>({1, 1, 1, 0, 0, 0}, {6});
  const auto ids = array<std::int32_t>({3, 4, 5}, {3});

  const auto by_mask = tf::cpp::reindexed_by_mask_on_points(mesh.mesh(), mask);
  const auto by_mask_maps =
      tf::cpp::reindexed_by_mask_on_points_with_maps(mesh.mesh(), mask);
  const auto by_ids = tf::cpp::reindexed_by_ids_on_points(mesh.mesh(), ids);
  const auto by_ids_maps =
      tf::cpp::reindexed_by_ids_on_points_with_maps(mesh.mesh(), ids);
  CHECK(by_mask.size() == 1);
  CHECK(by_mask_maps.face_map.kept_ids[0] == 0);
  CHECK(by_ids.size() == 1);
  CHECK(by_ids.points_buffer().data_buffer()[0] == Catch::Approx(TestType{10}));
  CHECK(by_ids_maps.face_map.kept_ids[0] == 1);

  auto empty = empty_mesh<TestType>();
  const auto no_mask = array<std::int8_t>({}, {0});
  const auto no_ids = array<std::int32_t>({}, {0});
  CHECK(tf::cpp::reindexed_by_mask(empty.mesh(), no_mask).size() == 0);
  CHECK(
      tf::cpp::reindexed_by_ids(empty.mesh(), no_ids).points_buffer().size() ==
      0);
  CHECK(tf::cpp::reindexed_by_mask_on_points(empty.mesh(), no_mask).size() ==
        0);
  CHECK(tf::cpp::reindexed_by_ids_on_points(empty.mesh(), no_ids)
            .points_buffer()
            .size() == 0);
}

TEMPLATE_TEST_CASE("reindexed validates and applies canonical maps",
                   "[cpp][reindex][maps]", float, double) {
  auto mesh = two_triangles<TestType>();
  const auto mask = array<std::int8_t>({0, 1}, {2});
  const auto filtered = tf::cpp::reindexed_by_mask_with_maps(mesh.mesh(), mask);
  const auto reapplied =
      tf::cpp::reindexed(mesh.mesh(), filtered.face_map, filtered.point_map);
  CHECK(equal_mesh(reapplied, filtered.mesh));

  auto wrong_size = filtered.face_map.deep_copy();
  wrong_size.f = array<std::int32_t>({0}, {1});
  CHECK_THROWS_AS(
      tf::cpp::reindexed(mesh.mesh(), wrong_size, filtered.point_map),
      std::invalid_argument);

  auto bad_value = filtered.face_map.deep_copy();
  bad_value.f[0] = 4;
  CHECK_THROWS_AS(
      tf::cpp::reindexed(mesh.mesh(), bad_value, filtered.point_map),
      std::out_of_range);

  auto inconsistent = filtered.point_map.deep_copy();
  inconsistent.kept_ids[0] = 4;
  CHECK_THROWS_AS(
      tf::cpp::reindexed(mesh.mesh(), filtered.face_map, inconsistent),
      std::invalid_argument);

  auto removes_used_point = filtered.point_map.deep_copy();
  removes_used_point.f[3] =
      static_cast<std::int32_t>(mesh.polygons.points_buffer().size());
  CHECK_THROWS_AS(
      tf::cpp::reindexed(mesh.mesh(), filtered.face_map, removes_used_point),
      std::invalid_argument);
}

TEMPLATE_TEST_CASE("reindexed accepts many-to-one clean maps",
                   "[cpp][reindex][maps][clean]", float, double) {
  auto source = duplicate_mesh<TestType>();
  auto cleaned = tf::cpp::cleaned_mesh_with_maps(source.mesh(), TestType{0.02},
                                                 true, false);

  const auto reapplied =
      tf::cpp::reindexed(source.mesh(), cleaned.face_map, cleaned.point_map);
  CHECK(equal_mesh(reapplied, cleaned.mesh));

  auto pending = tf::cpp::async::reindexed(source.mesh(), cleaned.face_map,
                                           cleaned.point_map);
  cleaned.face_map = {};
  cleaned.point_map = {};
  CHECK(equal_mesh(pending.get(), cleaned.mesh));
}

TEMPLATE_TEST_CASE("concatenate applies placements and split preserves labels",
                   "[cpp][reindex][concatenate][split]", float, double) {
  auto first = two_triangles<TestType>();
  auto second = two_triangles<TestType>();
  second.place(translation<TestType>(TestType{100}));
  const auto joined = tf::cpp::concatenate_meshes(
      std::vector<tf::cpp::mesh<tf::cpp::default_index_t, TestType>>{
          first.mesh(), second.mesh()});
  CHECK(joined.size() == 4);
  CHECK(joined.points_buffer().size() == 12);
  CHECK(joined.points_buffer().data_buffer()[18] ==
        Catch::Approx(TestType{100}));

  auto joined_owned = owned_t<TestType>{joined};
  const auto labels = array<std::int32_t>({7, 7, 3, 3}, {4});
  const auto split =
      tf::cpp::split_into_components(joined_owned.mesh(), labels);
  static_assert(
      std::is_same_v<decltype(split),
                     const tf::cpp::split_components_result<tf::polygons_buffer<
                         tf::cpp::default_index_t, TestType, 3, 3>>>);
  REQUIRE(split.components.size() == 2);
  CHECK(split.labels.raw_shape() == tf::small_vector<int, 3>{2});
  CHECK(split.labels[0] == 3);
  CHECK(split.labels[1] == 7);
  CHECK(split.components[0].size() == 2);
  CHECK(split.components[1].size() == 2);

  auto empty = empty_mesh<TestType>();
  const auto no_labels = array<std::int32_t>({}, {0});
  const auto empty_split =
      tf::cpp::split_into_components(empty.mesh(), no_labels);
  CHECK(empty_split.components.empty());
  CHECK(empty_split.labels.raw_shape() == tf::small_vector<int, 3>{0});
  CHECK_THROWS_AS(
      tf::cpp::concatenate_meshes(
          std::vector<tf::cpp::mesh<tf::cpp::default_index_t, TestType>>{}),
      std::invalid_argument);
}

TEMPLATE_TEST_CASE("split domains validates typed labels and retains them",
                   "[cpp][reindex][domains]", float, double) {
  auto mesh = two_triangles<TestType>();
  tf::cpp::domain_labels_result labels(
      array<std::int32_t>({0, 2, 1, 2}, {2, 2}), 2, -1);

  const auto split = tf::cpp::split_into_domains(mesh.mesh(), labels);
  REQUIRE(split.components.size() == 2);
  CHECK(split.labels.raw_shape() == tf::small_vector<int, 3>{2});
  CHECK(split.labels[0] == 0);
  CHECK(split.labels[1] == 1);
  CHECK(split.components[0].size() == 1);
  CHECK(split.components[1].size() == 1);

  auto pending = tf::cpp::async::split_into_domains(mesh.mesh(), labels);
  static_assert(std::is_same_v<decltype(pending),
                               std::future<tf::cpp::split_domains_result<
                                   tf::cpp::default_index_t, TestType>>>);
  CHECK(pending.get().components.size() == 2);

  auto valid_mesh = two_triangles<TestType>();
  const tf::cpp::domain_labels_result wrong_faces(
      array<std::int32_t>({0, 1}, {1, 2}), 2, -1);
  CHECK_THROWS_AS(tf::cpp::split_into_domains(valid_mesh.mesh(), wrong_faces),
                  std::invalid_argument);
  const tf::cpp::domain_labels_result missing_domain(
      array<std::int32_t>({0, 2, 0, 2}, {2, 2}), 2, -1);
  CHECK_THROWS_AS(
      tf::cpp::split_into_domains(valid_mesh.mesh(), missing_domain),
      std::invalid_argument);
  tf::cpp::domain_labels_result mutated_labels(
      array<std::int32_t>({0, 2, 1, 2}, {2, 2}), 2, -1);
  mutated_labels.labels()[0] = 3;
  CHECK_THROWS_AS(
      tf::cpp::split_into_domains(valid_mesh.mesh(), mutated_labels),
      std::out_of_range);

  auto empty = empty_mesh<TestType>();
  CHECK(
      tf::cpp::split_into_domains(empty.mesh(), tf::cpp::domain_labels_result{})
          .components.empty());
}

TEMPLATE_TEST_CASE("reindex rejects malformed selectors before kernel access",
                   "[cpp][reindex][validation]", float, double) {
  auto mesh = two_triangles<TestType>();
  const auto short_mask = array<std::int8_t>({1}, {1});
  const auto matrix_mask = array<std::int8_t>({1, 0}, {1, 2});
  const auto negative = array<std::int32_t>({-1}, {1});
  const auto too_large = array<std::int32_t>({2}, {1});
  const auto duplicate = array<std::int32_t>({0, 0}, {2});
  const auto short_point_mask = array<std::int8_t>({1}, {1});
  const auto short_labels = array<std::int32_t>({0}, {1});

  CHECK_THROWS_AS(tf::cpp::reindexed_by_mask(mesh.mesh(), short_mask),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::reindexed_by_mask(mesh.mesh(), matrix_mask),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::reindexed_by_ids(mesh.mesh(), negative),
                  std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::reindexed_by_ids(mesh.mesh(), too_large),
                  std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::reindexed_by_ids(mesh.mesh(), duplicate),
                  std::invalid_argument);
  CHECK_THROWS_AS(
      tf::cpp::reindexed_by_mask_on_points(mesh.mesh(), short_point_mask),
      std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::reindexed_by_ids_on_points(mesh.mesh(), negative),
                  std::out_of_range);
  CHECK_THROWS_AS(tf::cpp::split_into_components(mesh.mesh(), short_labels),
                  std::invalid_argument);
}

/// The cache owns the indices for the reading it answers, so every entry
/// refuses a corner that names a point the geometry does not have.
TEMPLATE_TEST_CASE("invalid face indices fail before every reindex kernel",
                   "[cpp][reindex][safety]", float, double) {
  const auto face_mask = array<std::int8_t>({1, 1}, {2});
  const auto point_mask = array<std::int8_t>({1, 1, 1, 1, 1, 1}, {6});
  const auto labels = array<std::int32_t>({0, 1}, {2});
  {
    auto negative = two_triangles<TestType>();
    negative.polygons.faces_buffer().data_buffer()[0] = -1;
    negative.cache.faces_changed();
    CHECK_THROWS_AS(tf::cpp::reindexed_by_mask(negative.mesh(), face_mask),
                    std::out_of_range);
    auto beyond = two_triangles<TestType>();
    beyond.polygons.faces_buffer().data_buffer()[0] = 6;
    beyond.cache.faces_changed();
    CHECK_THROWS_AS(tf::cpp::reindexed_by_mask(beyond.mesh(), face_mask),
                    std::out_of_range);
  }
  {
    auto mesh = two_triangles<TestType>();
    mesh.polygons.points_buffer().data_buffer().allocate(6);
    mesh.cache.points_changed();
    CHECK_THROWS_AS(tf::cpp::reindexed_by_mask(mesh.mesh(), face_mask),
                    std::out_of_range);
    CHECK_THROWS_AS(
        tf::cpp::reindexed_by_mask_on_points(mesh.mesh(), point_mask),
        std::out_of_range);
    CHECK_THROWS_AS(tf::cpp::split_into_components(mesh.mesh(), labels),
                    std::out_of_range);
  }
}

TEMPLATE_TEST_CASE("high-valence point maps are deterministic",
                   "[cpp][reindex][determinism]", float, double) {
  auto mesh = high_valence_fan<TestType>();
  const auto face_count = static_cast<int>(mesh.polygons.size());
  tf::buffer<std::int8_t> mask_buffer;
  mask_buffer.allocate(static_cast<std::size_t>(face_count));
  std::fill(mask_buffer.begin(), mask_buffer.end(), std::int8_t{1});
  const auto mask = tf::cpp::nd_array<std::int8_t>::from_buffer(
      std::move(mask_buffer), {face_count});
  const auto expected_map =
      tf::cpp::reindexed_by_mask_with_maps(mesh.mesh(), mask).point_map;

  for (int iteration = 0; iteration < 32; ++iteration) {
    mesh.cache.points_changed();
    const auto actual =
        tf::cpp::reindexed_by_mask_with_maps(mesh.mesh(), mask).point_map;
    CHECK(equal_array(actual.f, expected_map.f));
    CHECK(equal_array(actual.kept_ids, expected_map.kept_ids));
  }
}

/// An async selection carries its arrays as the handles they are and reads
/// the caller's own mesh, so the caller keeps both alive until the future
/// completes and releases rather than rewrites.
TEMPLATE_TEST_CASE("async reindex retains its arrays and preserves exact "
                   "results",
                   "[cpp][reindex][async]", float, double) {
  auto mesh = two_triangles<TestType>();
  auto mask = array<std::int8_t>({1, 0}, {2});
  auto ids = array<std::int32_t>({0}, {1});
  auto point_mask = array<std::int8_t>({1, 1, 1, 0, 0, 0}, {6});
  auto point_ids = array<std::int32_t>({0, 1, 2}, {3});
  auto labels = array<std::int32_t>({5, 9}, {2});
  const auto maps = tf::cpp::reindexed_by_mask_with_maps(mesh.mesh(), mask);

  auto reapplied =
      tf::cpp::async::reindexed(mesh.mesh(), maps.face_map, maps.point_map);
  auto by_mask = tf::cpp::async::reindexed_by_mask(mesh.mesh(), mask);
  auto by_mask_maps =
      tf::cpp::async::reindexed_by_mask_with_maps(mesh.mesh(), mask);
  auto by_ids = tf::cpp::async::reindexed_by_ids(mesh.mesh(), ids);
  auto by_ids_maps =
      tf::cpp::async::reindexed_by_ids_with_maps(mesh.mesh(), ids);
  auto by_point_mask =
      tf::cpp::async::reindexed_by_mask_on_points(mesh.mesh(), point_mask);
  auto by_point_mask_maps =
      tf::cpp::async::reindexed_by_mask_on_points_with_maps(mesh.mesh(),
                                                            point_mask);
  auto by_point_ids =
      tf::cpp::async::reindexed_by_ids_on_points(mesh.mesh(), point_ids);
  auto by_point_ids_maps = tf::cpp::async::reindexed_by_ids_on_points_with_maps(
      mesh.mesh(), point_ids);
  auto concatenated = tf::cpp::async::concatenate_meshes(
      std::vector<tf::cpp::mesh<tf::cpp::default_index_t, TestType>>{
          mesh.mesh(), mesh.mesh()});
  auto split = tf::cpp::async::split_into_components(mesh.mesh(), labels);

  static_assert(
      std::is_same_v<decltype(reapplied), std::future<selected_t<TestType>>>);
  static_assert(std::is_same_v<decltype(by_mask_maps),
                               std::future<tf::cpp::reindexed_mesh_result<
                                   tf::cpp::default_index_t, TestType>>>);
  static_assert(
      std::is_same_v<
          decltype(split),
          std::future<tf::cpp::split_components_result<selected_t<TestType>>>>);

  mask.destroy();
  ids.destroy();
  point_mask.destroy();
  point_ids.destroy();
  labels.destroy();

  CHECK(reapplied.get().size() == 1);
  CHECK(by_mask.get().size() == 1);
  CHECK(by_mask_maps.get().mesh.size() == 1);
  CHECK(by_ids.get().size() == 1);
  CHECK(by_ids_maps.get().mesh.size() == 1);
  CHECK(by_point_mask.get().size() == 1);
  CHECK(by_point_mask_maps.get().mesh.size() == 1);
  CHECK(by_point_ids.get().size() == 1);
  CHECK(by_point_ids_maps.get().mesh.size() == 1);
  CHECK(concatenated.get().size() == 4);
  CHECK(split.get().components.size() == 2);
}

TEST_CASE("the async mesh reindex takes the arity its sync twin takes",
          "[cpp][reindex][async]") {
  // reindexing a mixed mesh is what reindex is for, so the entry that answers
  // it on an executor is the same entry
  using mixed_result_t =
      tf::polygons_buffer<tf::cpp::default_index_t, float, 3, tf::dynamic_size>;
  tf::cpp::test::owned_mesh<tf::cpp::default_index_t, float, 3,
                            tf::dynamic_size>
      mesh{tf::cpp::test::polygons_of<tf::cpp::default_index_t, float>(
          {0, 3, 7}, {0, 1, 2, 3, 4, 5, 6},
          {0, 0, 0, 1, 0, 0, 0, 1, 0, 10, 0, 0, 11, 0, 0, 11, 1, 0, 10, 1, 0})};
  const auto maps = tf::cpp::reindexed_by_ids_with_maps(
      mesh.mesh(), array<std::int32_t>({1}, {1}));

  auto pending =
      tf::cpp::async::reindexed(mesh.mesh(), maps.face_map, maps.point_map);
  static_assert(std::is_same_v<decltype(pending), std::future<mixed_result_t>>);

  const auto result = pending.get();
  CHECK(result.size() == 1);
  CHECK(result.points_buffer().size() == 4);
  CHECK(equal_array(tf::cpp::test::face_indices_of(result),
                    array<std::int32_t>({0, 1, 2, 3}, {4})));
}

TEST_CASE("async reindex supports custom resolvers and exceptions",
          "[cpp][reindex][async][resolver]") {
  auto mesh = two_triangles<float>();
  auto ids = array<std::int32_t>({0}, {1});
  const auto submissions = std::make_shared<std::atomic<int>>(0);
  auto custom = tf::cpp::async::reindexed_by_ids(counting_resolver{submissions},
                                                 mesh.mesh(), ids);
  CHECK(submissions->load(std::memory_order_relaxed) == 1);
  CHECK(custom.get().size() == 1);

  auto invalid = array<std::int32_t>({-1}, {1});
  auto failure = tf::cpp::async::reindexed_by_ids(mesh.mesh(), invalid);
  CHECK_THROWS_AS(failure.get(), std::out_of_range);
}

TEMPLATE_TEST_CASE("reindex facade symbols link from the native archive",
                   "[cpp][reindex][archive-link]", float, double) {
  using mesh_t = tf::cpp::mesh<tf::cpp::default_index_t, TestType>;
  auto (*apply)(const mesh_t &, const tf::cpp::index_map<> &,
                const tf::cpp::index_map<> &)
      ->selected_t<TestType> =
      &tf::cpp::reindexed<std::int32_t, TestType, 3, 3>;
  auto (*mask)(const mesh_t &, const tf::cpp::nd_array<std::int8_t> &)
      ->selected_t<TestType> =
      &tf::cpp::reindexed_by_mask<std::int32_t, TestType, 3, 3>;
  auto (*ids)(const mesh_t &, const tf::cpp::nd_array<std::int32_t> &)
      ->selected_t<TestType> =
      &tf::cpp::reindexed_by_ids<std::int32_t, TestType, 3, 3>;
  auto (*point_mask)(const mesh_t &, const tf::cpp::nd_array<std::int8_t> &)
      ->selected_t<TestType> =
      &tf::cpp::reindexed_by_mask_on_points<std::int32_t, TestType, 3, 3>;
  auto (*point_ids)(const mesh_t &, const tf::cpp::nd_array<std::int32_t> &)
      ->selected_t<TestType> =
      &tf::cpp::reindexed_by_ids_on_points<std::int32_t, TestType, 3, 3>;
  auto (*concatenate)(const std::vector<mesh_t> &)->selected_t<TestType> =
      &tf::cpp::concatenate_meshes<std::int32_t, TestType, 3, 3>;
  auto (*split)(const mesh_t &, const tf::cpp::nd_array<std::int32_t> &)
      ->tf::cpp::split_components_result<selected_t<TestType>> =
      &tf::cpp::split_into_components<std::int32_t, TestType, 3, 3>;

  auto mesh = two_triangles<TestType>();
  const auto selection = array<std::int8_t>({1, 0}, {2});
  const auto selected =
      tf::cpp::reindexed_by_mask_with_maps(mesh.mesh(), selection);
  CHECK(apply(mesh.mesh(), selected.face_map, selected.point_map).size() == 1);
  CHECK(mask(mesh.mesh(), selection).size() == 1);
  CHECK(ids(mesh.mesh(), array<std::int32_t>({0}, {1})).size() == 1);
  CHECK(point_mask(mesh.mesh(), array<std::int8_t>({1, 1, 1, 0, 0, 0}, {6}))
            .size() == 1);
  CHECK(point_ids(mesh.mesh(), array<std::int32_t>({0, 1, 2}, {3})).size() ==
        1);
  CHECK(concatenate({mesh.mesh(), mesh.mesh()}).size() == 4);
  CHECK(
      split(mesh.mesh(), array<std::int32_t>({0, 1}, {2})).components.size() ==
      2);
}

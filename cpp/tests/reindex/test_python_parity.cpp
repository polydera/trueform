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

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename Real>
using parity_owned = tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real>;

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

template <typename T>
auto has_values(const tf::cpp::nd_array<T> &value,
                std::initializer_list<T> expected) -> bool {
  if (value.length() != expected.size())
    return false;
  auto actual = value.begin();
  for (const auto item : expected)
    if (*actual++ != item)
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

template <typename Real> auto selection_mesh() -> parity_owned<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 1, 3, 2, 3, 4, 2},
      {0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 2, 1, 0})};
}

template <typename Real> auto component_mesh() -> parity_owned<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 1, 3, 2, 3, 4, 5, 3, 5, 4},
      {0, 0, 0, 1, 0, 0, Real{0.5}, 1, 0, 2, 0, 0, Real{2.5}, 1, 0, 3, 0, 0})};
}

template <typename Real>
auto translation(Real x, Real y, Real z) -> std::array<Real, 16> {
  return {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, Real{1}};
}

} // namespace

TEMPLATE_TEST_CASE(
    "Python-parity face IDs preserve requested order and exact maps",
    "[cpp][reindex][python-parity][ids]", float, double) {
  auto input = selection_mesh<TestType>();
  const auto original_faces = tf::cpp::test::face_indices_of(input.polygons);
  const auto original_points = points_array(input.polygons.points_buffer());
  const auto ids = make_array<std::int32_t>({2, 0}, {2});

  auto result = tf::cpp::reindexed_by_ids_with_maps(input.mesh(), ids);
  CHECK(result.mesh.size() == 2);
  CHECK(result.mesh.points_buffer().size() == 5);
  CHECK(has_values(tf::cpp::test::face_indices_of(result.mesh),
                   {std::int32_t{3}, std::int32_t{4}, std::int32_t{2},
                    std::int32_t{0}, std::int32_t{1}, std::int32_t{2}}));
  CHECK(has_values(result.face_map.f,
                   {std::int32_t{1}, std::int32_t{3}, std::int32_t{0}}));
  CHECK(
      has_values(result.face_map.kept_ids, {std::int32_t{2}, std::int32_t{0}}));
  CHECK(has_values(result.point_map.f,
                   {std::int32_t{0}, std::int32_t{1}, std::int32_t{2},
                    std::int32_t{3}, std::int32_t{4}}));
  CHECK(has_values(result.point_map.kept_ids,
                   {std::int32_t{0}, std::int32_t{1}, std::int32_t{2},
                    std::int32_t{3}, std::int32_t{4}}));
  CHECK(same_array(points_array(result.mesh.points_buffer()), original_points));

  CHECK(same_array(tf::cpp::test::face_indices_of(input.polygons),
                   original_faces));
  CHECK(same_array(points_array(input.polygons.points_buffer()),
                   original_points));
  result.mesh.points_buffer().data_buffer()[0] = TestType{99};
  CHECK(input.polygons.points_buffer().data_buffer()[0] == TestType{0});

  const auto repeated = tf::cpp::reindexed_by_ids_with_maps(input.mesh(), ids);
  CHECK_FALSE(same_array(points_array(repeated.mesh.points_buffer()),
                         points_array(result.mesh.points_buffer())));
  result.mesh.points_buffer().data_buffer()[0] = TestType{0};
  CHECK(same_array(tf::cpp::test::face_indices_of(repeated.mesh),
                   tf::cpp::test::face_indices_of(result.mesh)));
  CHECK(same_array(points_array(repeated.mesh.points_buffer()),
                   points_array(result.mesh.points_buffer())));
  CHECK(same_array(repeated.face_map.f, result.face_map.f));
  CHECK(same_array(repeated.point_map.f, result.point_map.f));
}

TEST_CASE("Python-parity reindex selectors and split labels validate rank",
          "[cpp][reindex][python-parity][validation]") {
  auto input = selection_mesh<float>();
  const auto matrix_ids = make_array<std::int32_t>({0}, {1, 1});
  const auto matrix_labels = make_array<std::int32_t>({0, 1, 2}, {1, 3});

  CHECK_THROWS_AS(tf::cpp::reindexed_by_ids(input.mesh(), matrix_ids),
                  std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::split_into_components(input.mesh(), matrix_labels),
                  std::invalid_argument);
}

TEMPLATE_TEST_CASE("Python-parity point IDs compact connectivity and maps",
                   "[cpp][reindex][python-parity][point-ids]", float, double) {
  auto input = selection_mesh<TestType>();
  const auto ids = make_array<std::int32_t>({2, 3, 4}, {3});
  const auto result =
      tf::cpp::reindexed_by_ids_on_points_with_maps(input.mesh(), ids);

  CHECK(has_values(tf::cpp::test::face_indices_of(result.mesh),
                   {std::int32_t{1}, std::int32_t{2}, std::int32_t{0}}));
  CHECK(has_values(points_array(result.mesh.points_buffer()),
                   {TestType{0}, TestType{1}, TestType{0}, TestType{2},
                    TestType{0}, TestType{0}, TestType{2}, TestType{1},
                    TestType{0}}));
  CHECK(has_values(result.face_map.f,
                   {std::int32_t{3}, std::int32_t{3}, std::int32_t{0}}));
  CHECK(has_values(result.face_map.kept_ids, {std::int32_t{2}}));
  CHECK(has_values(result.point_map.f,
                   {std::int32_t{5}, std::int32_t{5}, std::int32_t{0},
                    std::int32_t{1}, std::int32_t{2}}));
  CHECK(has_values(result.point_map.kept_ids,
                   {std::int32_t{2}, std::int32_t{3}, std::int32_t{4}}));
}

TEMPLATE_TEST_CASE(
    "Python-parity concatenation rebases faces and applies placements",
    "[cpp][reindex][python-parity][concatenate]", float, double) {
  parity_owned<TestType> first{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, 2, 1, 2, 3}, {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0})};
  parity_owned<TestType> second{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, 2}, {0, 0, 0, 1, 0, 0, 0, 1, 0})};
  second.place(translation<TestType>(TestType{10}, TestType{2}, TestType{3}));
  const auto first_faces = tf::cpp::test::face_indices_of(first.polygons);
  const auto first_points = points_array(first.polygons.points_buffer());
  const auto second_points = points_array(second.polygons.points_buffer());

  const auto result = tf::cpp::concatenate_meshes(
      std::vector<tf::cpp::mesh<tf::cpp::default_index_t, TestType>>{
          first.mesh(), second.mesh()});
  CHECK(has_values(tf::cpp::test::face_indices_of(result),
                   {std::int32_t{0}, std::int32_t{1}, std::int32_t{2},
                    std::int32_t{1}, std::int32_t{2}, std::int32_t{3},
                    std::int32_t{4}, std::int32_t{5}, std::int32_t{6}}));
  CHECK(has_values(points_array(result.points_buffer()),
                   {TestType{0},  TestType{0}, TestType{0},  TestType{1},
                    TestType{0},  TestType{0}, TestType{1},  TestType{1},
                    TestType{0},  TestType{0}, TestType{1},  TestType{0},
                    TestType{10}, TestType{2}, TestType{3},  TestType{11},
                    TestType{2},  TestType{3}, TestType{10}, TestType{3},
                    TestType{3}}));
  CHECK(same_array(tf::cpp::test::face_indices_of(first.polygons),
                   first_faces));
  CHECK(same_array(points_array(first.polygons.points_buffer()), first_points));
  CHECK(
      same_array(points_array(second.polygons.points_buffer()), second_points));

  const auto singleton = tf::cpp::concatenate_meshes(
      std::vector<tf::cpp::mesh<tf::cpp::default_index_t, TestType>>{
          first.mesh()});
  CHECK(same_array(tf::cpp::test::face_indices_of(singleton), first_faces));
  CHECK(same_array(points_array(singleton.points_buffer()), first_points));
}

TEMPLATE_TEST_CASE(
    "Python-parity component splitting sorts labels and compacts points",
    "[cpp][reindex][python-parity][split]", float, double) {
  auto input = component_mesh<TestType>();
  const auto original_faces = tf::cpp::test::face_indices_of(input.polygons);
  const auto original_points = points_array(input.polygons.points_buffer());
  const auto labels = make_array<std::int32_t>({2, 0, 1, 1}, {4});

  const auto result = tf::cpp::split_into_components(input.mesh(), labels);
  REQUIRE(result.components.size() == 3);
  CHECK(has_values(result.labels,
                   {std::int32_t{0}, std::int32_t{1}, std::int32_t{2}}));

  CHECK(has_values(tf::cpp::test::face_indices_of(result.components[0]),
                   {std::int32_t{0}, std::int32_t{1}, std::int32_t{2}}));
  CHECK(has_values(points_array(result.components[0].points_buffer()),
                   {TestType{1}, TestType{0}, TestType{0}, TestType{2},
                    TestType{0}, TestType{0}, TestType{0.5}, TestType{1},
                    TestType{0}}));
  CHECK(has_values(tf::cpp::test::face_indices_of(result.components[1]),
                   {std::int32_t{0}, std::int32_t{1}, std::int32_t{2},
                    std::int32_t{0}, std::int32_t{2}, std::int32_t{1}}));
  CHECK(has_values(points_array(result.components[1].points_buffer()),
                   {TestType{2}, TestType{0}, TestType{0}, TestType{2.5},
                    TestType{1}, TestType{0}, TestType{3}, TestType{0},
                    TestType{0}}));
  CHECK(has_values(tf::cpp::test::face_indices_of(result.components[2]),
                   {std::int32_t{0}, std::int32_t{1}, std::int32_t{2}}));
  CHECK(has_values(points_array(result.components[2].points_buffer()),
                   {TestType{0}, TestType{0}, TestType{0}, TestType{1},
                    TestType{0}, TestType{0}, TestType{0.5}, TestType{1},
                    TestType{0}}));

  CHECK(same_array(tf::cpp::test::face_indices_of(input.polygons),
                   original_faces));
  CHECK(same_array(points_array(input.polygons.points_buffer()),
                   original_points));

  const auto repeated = tf::cpp::split_into_components(input.mesh(), labels);
  REQUIRE(repeated.components.size() == result.components.size());
  CHECK(same_array(repeated.labels, result.labels));
  for (std::size_t index = 0; index < result.components.size(); ++index) {
    CHECK(same_array(tf::cpp::test::face_indices_of(repeated.components[index]),
                     tf::cpp::test::face_indices_of(result.components[index])));
    CHECK(same_array(points_array(repeated.components[index].points_buffer()),
                     points_array(result.components[index].points_buffer())));
  }
}

TEMPLATE_TEST_CASE("Python-parity domain splitting has stable domain order",
                   "[cpp][reindex][python-parity][split-domains]", float,
                   double) {
  parity_owned<TestType> input{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, 2, 3, 4, 5},
          {0, 0, 0, 1, 0, 0, 0, 1, 0, 10, 0, 0, 11, 0, 0, 10, 1, 0})};
  const tf::cpp::domain_labels_result labels(
      make_array<std::int32_t>({0, 2, 1, 2}, {2, 2}), 2, -1);

  const auto result = tf::cpp::split_into_domains(input.mesh(), labels);
  REQUIRE(result.components.size() == 2);
  CHECK(has_values(result.labels, {std::int32_t{0}, std::int32_t{1}}));
  CHECK(has_values(tf::cpp::test::face_indices_of(result.components[0]),
                   {std::int32_t{2}, std::int32_t{1}, std::int32_t{0}}));
  CHECK(has_values(points_array(result.components[0].points_buffer()),
                   {TestType{0}, TestType{0}, TestType{0}, TestType{1},
                    TestType{0}, TestType{0}, TestType{0}, TestType{1},
                    TestType{0}}));
  CHECK(has_values(tf::cpp::test::face_indices_of(result.components[1]),
                   {std::int32_t{2}, std::int32_t{1}, std::int32_t{0}}));
  CHECK(has_values(points_array(result.components[1].points_buffer()),
                   {TestType{10}, TestType{0}, TestType{0}, TestType{11},
                    TestType{0}, TestType{0}, TestType{10}, TestType{1},
                    TestType{0}}));
}

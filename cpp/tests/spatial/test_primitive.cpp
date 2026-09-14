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
#include "trueform/cpp/spatial/primitive.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct primitive_case {
  tf::cpp::primitive_kind kind;
  tf::small_vector<int, 3> single_shape;
  tf::small_vector<int, 3> batch_shape;
  int polygon_vertex_count;
  std::size_t element_stride;
};

auto primitive_cases() -> std::vector<primitive_case> {
  using tf::cpp::primitive_kind;
  return {
      {primitive_kind::point, {3}, {2, 3}, 0, 3},
      {primitive_kind::vector, {3}, {2, 3}, 0, 3},
      {primitive_kind::segment, {2, 3}, {2, 2, 3}, 0, 6},
      {primitive_kind::triangle, {3, 3}, {2, 3, 3}, 0, 9},
      {primitive_kind::ray, {2, 3}, {2, 2, 3}, 0, 6},
      {primitive_kind::line, {2, 3}, {2, 2, 3}, 0, 6},
      {primitive_kind::plane, {4}, {2, 4}, 0, 4},
      {primitive_kind::aabb, {2, 3}, {2, 2, 3}, 0, 6},
      {primitive_kind::polygon, {4, 3}, {2, 4, 3}, 4, 12},
  };
}

template <typename T>
auto make_array(tf::small_vector<int, 3> shape) -> tf::cpp::nd_array<T> {
  std::size_t length = 1;
  for (const auto dimension : shape)
    length *= static_cast<std::size_t>(dimension);

  tf::buffer<T> buffer;
  buffer.allocate(length);
  for (std::size_t index = 0; index < length; ++index)
    buffer[index] = static_cast<T>(index + 1);
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

template <typename T>
auto require_shape(const tf::cpp::nd_array<T> &array,
                   const tf::small_vector<int, 3> &shape) -> void {
  REQUIRE(array.ndim() == static_cast<int>(shape.size()));
  for (int dimension = 0; dimension < array.ndim(); ++dimension)
    CHECK(array.shape_at(dimension) ==
          shape[static_cast<std::size_t>(dimension)]);
}

auto make_lifetime_view() -> tf::cpp::primitive<float> {
  auto batch = tf::cpp::primitive<float>(tf::cpp::primitive_kind::point,
                                         make_array<float>({2, 3}));
  return batch.at(1);
}

} // namespace

TEMPLATE_TEST_CASE(
    "3D primitives accept every canonical single and batch shape",
    "[cpp][spatial][primitive]", float, double) {
  for (const auto &spec : primitive_cases()) {
    INFO("primitive kind " << static_cast<int>(spec.kind));

    const auto single = tf::cpp::primitive<TestType>(
        spec.kind, make_array<TestType>(spec.single_shape));
    CHECK(single.kind() == spec.kind);
    CHECK(single.cardinality() == tf::cpp::primitive_cardinality::single);
    CHECK_FALSE(single.is_batch());
    CHECK(single.count() == 1);
    CHECK(single.polygon_vertex_count() == spec.polygon_vertex_count);
    CHECK(single.element_stride() == spec.element_stride);
    require_shape(single.data(), spec.single_shape);

    const auto batch = tf::cpp::primitive<TestType>(
        spec.kind, make_array<TestType>(spec.batch_shape));
    CHECK(batch.kind() == spec.kind);
    CHECK(batch.cardinality() == tf::cpp::primitive_cardinality::batch);
    CHECK(batch.is_batch());
    CHECK(batch.count() == 2);
    CHECK(batch.polygon_vertex_count() == spec.polygon_vertex_count);
    CHECK(batch.element_stride() == spec.element_stride);
    require_shape(batch.data(), spec.batch_shape);

    const auto element = batch.at(1);
    CHECK_FALSE(element.is_batch());
    CHECK(element.count() == 1);
    CHECK(element.kind() == spec.kind);
    CHECK(element.element_stride() == spec.element_stride);
    require_shape(element.data(), spec.single_shape);
    CHECK(element.data().raw_data() ==
          batch.data().raw_data() + spec.element_stride);
  }
}

TEST_CASE("primitive rank preserves empty and batch-of-one cardinality",
          "[cpp][spatial][primitive]") {
  for (const auto &spec : primitive_cases()) {
    INFO("primitive kind " << static_cast<int>(spec.kind));

    auto one_shape = spec.batch_shape;
    one_shape[0] = 1;
    const auto one = tf::cpp::primitive<float>(
        spec.kind, make_array<float>(std::move(one_shape)));
    CHECK(one.is_batch());
    CHECK(one.count() == 1);
    CHECK_FALSE(one.at(0).is_batch());

    auto empty_shape = spec.batch_shape;
    empty_shape[0] = 0;
    const auto empty = tf::cpp::primitive<float>(
        spec.kind, make_array<float>(std::move(empty_shape)));
    CHECK(empty.is_batch());
    CHECK(empty.count() == 0);
    CHECK(empty.data().empty());
    CHECK_THROWS_AS(empty.at(0), std::out_of_range);
  }
}

TEST_CASE("primitives reject 2D and malformed layouts",
          "[cpp][spatial][primitive]") {
  using tf::cpp::primitive;
  using tf::cpp::primitive_kind;

  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::point, make_array<float>({2})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::vector, make_array<float>({2})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::segment, make_array<float>({2, 2})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::triangle, make_array<float>({3, 2})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::ray, make_array<float>({2, 2})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::line, make_array<float>({2, 2})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::plane, make_array<float>({3})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::aabb, make_array<float>({2, 2})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::polygon, make_array<float>({3, 2})),
      std::invalid_argument);

  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::point, make_array<float>({1, 1, 3})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::segment, make_array<float>({6})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::triangle, make_array<float>({9})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::plane, make_array<float>({1, 1, 4})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::polygon, make_array<float>({9})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::polygon, make_array<float>({2, 3})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::polygon, make_array<float>({1, 2, 3})),
      std::invalid_argument);
  CHECK_THROWS_AS(
      primitive<float>(primitive_kind::point, tf::cpp::nd_array<float>{}),
      std::invalid_argument);
  CHECK_THROWS_AS(primitive<float>(static_cast<primitive_kind>(999),
                                   make_array<float>({3})),
                  std::invalid_argument);
}

TEST_CASE("primitive views retain storage and copies preserve ownership rules",
          "[cpp][spatial][primitive]") {
  auto value = tf::cpp::primitive<double>(tf::cpp::primitive_kind::vector,
                                          make_array<double>({2, 3}));
  auto shallow = value.shallow_copy();
  auto deep = value.deep_copy();

  shallow.data()[0] = 42;
  deep.data()[1] = 77;
  CHECK(value.data()[0] == 42);
  CHECK(value.data()[1] == 2);
  CHECK(deep.data()[1] == 77);
  CHECK(shallow.data().raw_data() == value.data().raw_data());
  CHECK(deep.data().raw_data() != value.data().raw_data());

  auto metadata_view = value.data();
  metadata_view.set_shape({6});
  CHECK(value.data().ndim() == 2);
  CHECK(value.data().shape_at(0) == 2);

  auto element = value.at(1);
  element.data()[0] = 99;
  CHECK(value.data()[value.element_stride()] == 99);

  const auto single = make_lifetime_view();
  REQUIRE(single.data().is_valid());
  CHECK(single.data()[0] == 4);
  CHECK(single.at(0).data().raw_data() == single.data().raw_data());
  CHECK_THROWS_AS(single.at(-1), std::out_of_range);
  CHECK_THROWS_AS(single.at(1), std::out_of_range);
}

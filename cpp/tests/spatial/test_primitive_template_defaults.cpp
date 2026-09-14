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
#include "trueform/cpp/spatial/distance.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

template <typename T> using array_storage_t = tf::buffer<T>;

template <typename T>
using array_shape_t = std::decay_t<
    decltype(std::declval<const tf::cpp::nd_array<T> &>().raw_shape())>;

template <typename Real>
auto make_array(std::initializer_list<Real> values, array_shape_t<Real> shape)
    -> tf::cpp::nd_array<Real> {
  array_storage_t<Real> buffer;
  buffer.allocate(values.size());
  auto output = buffer.begin();
  for (const auto value : values)
    *output++ = value;
  return tf::cpp::nd_array<Real>::from_buffer(std::move(buffer),
                                              std::move(shape));
}

template <typename Real, std::size_t Dims>
auto check_primitive_carrier_matrix_combination() -> void {
  INFO("Real bytes = " << sizeof(Real) << ", Dims = " << Dims);

  auto data = [&] {
    if constexpr (Dims == 2)
      return make_array<Real>({0, 0, 3, 4}, {2, 2});
    else
      return make_array<Real>({0, 0, 0, 3, 4, 5}, {2, 3});
  }();
  auto value = tf::cpp::primitive<Real, Dims>(tf::cpp::primitive_kind::point,
                                              std::move(data));

  STATIC_REQUIRE(
      (std::is_same_v<decltype(value.at(0)), tf::cpp::primitive<Real, Dims>>));
  STATIC_REQUIRE((std::is_same_v<decltype(value.slice(0, 1)),
                                 tf::cpp::primitive<Real, Dims>>));
  REQUIRE(value.kind() == tf::cpp::primitive_kind::point);
  REQUIRE(value.cardinality() == tf::cpp::primitive_cardinality::batch);
  CHECK(value.count() == 2);
  CHECK(value.element_stride() == Dims);
  CHECK(value.data().shape_at(0) == 2);
  CHECK(value.data().shape_at(1) == static_cast<int>(Dims));

  auto second = value.at(1);
  auto first_slice = value.slice(0, 1);
  REQUIRE_FALSE(second.is_batch());
  CHECK(second.count() == 1);
  CHECK(second.data().length() == Dims);
  CHECK(second.data()[0] == Real{3});
  REQUIRE(first_slice.is_batch());
  CHECK(first_slice.count() == 1);
  CHECK(first_slice.data().shape_at(0) == 1);
  CHECK(first_slice.data().shape_at(1) == static_cast<int>(Dims));

  auto shallow = value.shallow_copy();
  auto deep = value.deep_copy();
  CHECK(shallow.data().raw_data() == value.data().raw_data());
  CHECK(deep.data().raw_data() != value.data().raw_data());
  shallow.data()[0] = Real{7};
  deep.data()[1] = Real{9};
  CHECK(value.data()[0] == Real{7});
  CHECK(value.data()[1] == Real{0});
  CHECK(deep.data()[1] == Real{9});

  if constexpr (Dims == 2) {
    CHECK_THROWS_AS(
        (tf::cpp::primitive<Real, Dims>(tf::cpp::primitive_kind::plane,
                                        make_array<Real>({0, 1, 0}, {3}))),
        std::invalid_argument);
  } else {
    const auto plane = tf::cpp::primitive<Real, Dims>(
        tf::cpp::primitive_kind::plane, make_array<Real>({0, 0, 1, 0}, {4}));
    CHECK(plane.element_stride() == 4);
  }
}

} // namespace

TEST_CASE("approved primitive carrier matrix links with dimensional access",
          "[cpp][spatial][primitive][template][matrix]") {
  check_primitive_carrier_matrix_combination<float, 2>();
  check_primitive_carrier_matrix_combination<float, 3>();
  check_primitive_carrier_matrix_combination<double, 2>();
  check_primitive_carrier_matrix_combination<double, 3>();
}

TEST_CASE("an unstated primitive dimension is three, and every read keeps it",
          "[cpp][spatial][primitive][template]") {
  using unstated = tf::cpp::primitive<float>;
  using explicit_3d = tf::cpp::primitive<float, 3>;
  using explicit_2d = tf::cpp::primitive<float, 2>;

  STATIC_REQUIRE(std::is_same_v<unstated, explicit_3d>);
  STATIC_REQUIRE_FALSE(std::is_same_v<explicit_2d, explicit_3d>);
  STATIC_REQUIRE(std::is_same_v<decltype(std::declval<const unstated &>().at(0)),
                                explicit_3d>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(std::declval<const unstated &>().slice(0, 1)),
                     explicit_3d>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(std::declval<const unstated &>().shallow_copy()),
                     explicit_3d>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(std::declval<const unstated &>().deep_copy()),
                     explicit_3d>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(std::declval<const explicit_2d &>().at(0)),
                     explicit_2d>);
}

TEST_CASE("a primitive batch owns its storage and dispatches on its kind",
          "[cpp][spatial][primitive][template]") {
  auto points =
      tf::cpp::primitive<float>(tf::cpp::primitive_kind::point,
                                make_array<float>({0, 0, 0, 0, 3, 4}, {2, 3}));
  const auto origin = tf::cpp::primitive<double, 3>(
      tf::cpp::primitive_kind::point, make_array<double>({0, 0, 0}, {3}));

  REQUIRE(points.kind() == tf::cpp::primitive_kind::point);
  REQUIRE(points.cardinality() == tf::cpp::primitive_cardinality::batch);
  REQUIRE(points.is_batch());
  REQUIRE(points.count() == 2);
  REQUIRE(points.element_stride() == 3);

  const auto distances = tf::cpp::distance(points, origin);
  REQUIRE(distances.is_batch());
  REQUIRE(distances.batch().length() == 2);
  CHECK(distances.batch()[0] == Catch::Approx(0));
  CHECK(distances.batch()[1] == Catch::Approx(5));

  auto second = points.at(1);
  auto shallow = points.shallow_copy();
  auto deep = points.deep_copy();
  REQUIRE(second.data().raw_data() ==
          points.data().raw_data() + points.element_stride());
  REQUIRE(shallow.data().raw_data() == points.data().raw_data());
  REQUIRE(deep.data().raw_data() != points.data().raw_data());

  second.data()[0] = 7;
  shallow.data()[1] = 9;
  deep.data()[2] = 11;
  CHECK(points.data()[points.element_stride()] == 7);
  CHECK(points.data()[1] == 9);
  CHECK(points.data()[2] == 0);
  CHECK(deep.data()[2] == 11);
}

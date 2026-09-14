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
#include "nd_array.hpp"
#include "parity_checks.hpp"

#include "trueform/cpp/iso/async/isocontours.hpp"
#include "trueform/cpp/iso/isocontours.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <future>
#include <stdexcept>
#include <type_traits>

namespace {

template <typename Real>
using isocontour_owned =
    tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real>;

template <typename Real> auto isocontour_square() -> isocontour_owned<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 0, 2, 3},
      {Real{-1}, Real{-1}, Real{0}, Real{1}, Real{-1}, Real{0}, Real{1},
       Real{1}, Real{0}, Real{-1}, Real{1}, Real{0}})};
}

template <typename Real> auto isocontour_scalars() -> tf::cpp::nd_array<Real> {
  return tf::cpp::test::make_nd_array<Real>({-1, 1, 1, -1}, {4});
}

} // namespace

TEMPLATE_TEST_CASE("isocontours preserve scalar and threshold contracts",
                   "[cpp][iso][isocontours][sync]", float, double) {
  auto owner = isocontour_square<TestType>();
  const auto scalars = isocontour_scalars<TestType>();
  const auto cuts =
      tf::cpp::test::make_nd_array<TestType>({-0.5, 0, 0.5, 0}, {4});

  const auto single = tf::cpp::isocontours(owner.mesh(), scalars, TestType{0});
  const auto multi = tf::cpp::isocontours(owner.mesh(), scalars, cuts);
  CHECK(tf::cpp::test::curves_are_aligned(single, false));
  CHECK(single.size() > 0);
  CHECK(tf::cpp::test::curves_are_aligned(multi, false));
  CHECK(multi.size() >= single.size());

  // a contour is the operand's own, so it is read where it was authored
  owner.place({TestType{1}, TestType{0}, TestType{0}, TestType{10}, TestType{0},
               TestType{1}, TestType{0}, TestType{0}, TestType{0}, TestType{0},
               TestType{1}, TestType{0}, TestType{0}, TestType{0}, TestType{0},
               TestType{1}});
  const auto transformed =
      tf::cpp::isocontours(owner.mesh(), scalars, TestType{0});
  CHECK(
      tf::cpp::test::same_array(tf::cpp::test::arrays_of(single).points,
                                tf::cpp::test::arrays_of(transformed).points));

  const auto no_cuts = tf::cpp::test::make_nd_array<TestType>({}, {0});
  const auto empty_result =
      tf::cpp::isocontours(owner.mesh(), scalars, no_cuts);
  CHECK(tf::cpp::test::curves_are_aligned(empty_result, false));
  CHECK(empty_result.size() == 0);
}

TEMPLATE_TEST_CASE("isocontours validate scalars cuts and reachable faces",
                   "[cpp][iso][isocontours][validation]", float, double) {
  const auto square = isocontour_square<TestType>();
  const auto wrong_scalars =
      tf::cpp::test::make_nd_array<TestType>({0, 1, 2}, {3});
  const auto matrix_cuts =
      tf::cpp::test::make_nd_array<TestType>({0, 1}, {1, 2});
  CHECK_THROWS_AS(
      tf::cpp::isocontours(square.mesh(), wrong_scalars, TestType{0}),
      std::invalid_argument);
  CHECK_THROWS_AS(tf::cpp::isocontours(square.mesh(),
                                       isocontour_scalars<TestType>(),
                                       matrix_cuts),
                  std::invalid_argument);

  // faces that name a point the geometry does not have are refused at the one
  // door the reading passes
  const isocontour_owned<TestType> malformed{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, 2}, {TestType{0}, TestType{0}, TestType{0}, TestType{1},
                      TestType{0}, TestType{0}})};
  CHECK_THROWS_AS(
      tf::cpp::isocontours(malformed.mesh(),
                           tf::cpp::test::make_nd_array<TestType>({0, 1}, {2}),
                           TestType{0}),
      std::out_of_range);
}

TEMPLATE_TEST_CASE("isocontours borrow the mesh and copy the field arrays",
                   "[cpp][iso][isocontours][async]", float, double) {
  const auto square = isocontour_square<TestType>();
  auto scalars = isocontour_scalars<TestType>();
  auto cuts = tf::cpp::test::make_nd_array<TestType>({-0.5, 0.5}, {2});

  auto single_future =
      tf::cpp::async::isocontours(square.mesh(), scalars, TestType{0});
  auto multi_future = tf::cpp::async::isocontours(square.mesh(), scalars, cuts);
  static_assert(
      std::is_same_v<decltype(single_future),
                     std::future<tf::curves_buffer<tf::cpp::default_index_t,
                                                   TestType, 3>>>);
  static_assert(
      std::is_same_v<decltype(multi_future),
                     std::future<tf::curves_buffer<tf::cpp::default_index_t,
                                                   TestType, 3>>>);

  // the scalars are the call's copy, the mesh the caller's own reading
  scalars.destroy();
  cuts.destroy();

  CHECK(single_future.get().size() > 0);
  CHECK(multi_future.get().size() > 0);
}

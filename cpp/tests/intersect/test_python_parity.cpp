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
#include "canonicalize.hpp"
#include "carriers.hpp"
#include "fixtures.hpp"
#include "parity_checks.hpp"

#include "trueform/cpp/intersect/async/intersection_curves.hpp"
#include "trueform/cpp/intersect/async/self_intersection_curves.hpp"
#include "trueform/cpp/intersect/intersection_curves.hpp"
#include "trueform/cpp/intersect/self_intersection_curves.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <stdexcept>

namespace {

template <typename Real>
using parity_owned = tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real>;

/// A translation along x, as the sixteen numbers a placement is.
template <typename Real>
auto parity_x_translation(Real x) -> std::array<Real, 16> {
  return {Real{1}, Real{0}, Real{0}, x,       Real{0}, Real{1},
          Real{0}, Real{0}, Real{0}, Real{0}, Real{1}, Real{0},
          Real{0}, Real{0}, Real{0}, Real{1}};
}

template <typename Real>
auto horizontal_triangle(Real local_x_offset = Real{0}) -> parity_owned<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2},
      {Real{-1} + local_x_offset, Real{-1}, Real{0}, Real{1} + local_x_offset,
       Real{-1}, Real{0}, local_x_offset, Real{1}, Real{0}})};
}

template <typename Real> auto vertical_triangle() -> parity_owned<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2}, {Real{0}, Real{-0.5}, Real{-1}, Real{0}, Real{-0.5}, Real{1},
                  Real{0}, Real{0.75}, Real{0}})};
}

template <typename Real> auto crossing_triangles() -> parity_owned<Real> {
  return {tf::cpp::test::polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 3, 4, 5},
      {Real{-1}, Real{-1}, Real{0}, Real{1}, Real{-1}, Real{0}, Real{0},
       Real{1}, Real{0}, Real{0}, Real{-0.5}, Real{-1}, Real{0}, Real{-0.5},
       Real{1}, Real{0}, Real{0.75}, Real{0}})};
}

} // namespace

TEMPLATE_TEST_CASE(
    "Python-parity intersection curves are symmetric and match self curves",
    "[cpp][intersect][python-parity][curves]", float, double) {
  const auto horizontal = horizontal_triangle<TestType>();
  const auto vertical = vertical_triangle<TestType>();
  const auto combined = crossing_triangles<TestType>();
  const auto horizontal_faces =
      tf::cpp::test::parity_faces(horizontal.polygons).deep_copy();
  const auto horizontal_points =
      tf::cpp::test::parity_points(horizontal.polygons).deep_copy();
  const auto vertical_faces =
      tf::cpp::test::parity_faces(vertical.polygons).deep_copy();
  const auto vertical_points =
      tf::cpp::test::parity_points(vertical.polygons).deep_copy();
  const auto combined_faces =
      tf::cpp::test::parity_faces(combined.polygons).deep_copy();
  const auto combined_points =
      tf::cpp::test::parity_points(combined.polygons).deep_copy();
  const auto config = tf::intersect_config{tf::intersect_mode::primitives, 0.0};

  const auto forward =
      tf::cpp::intersection_curves(horizontal.mesh(), vertical.mesh(), config);
  const auto reverse =
      tf::cpp::intersection_curves(vertical.mesh(), horizontal.mesh(), config);
  const auto self = tf::cpp::self_intersection_curves(combined.mesh(), config);

  REQUIRE(tf::cpp::test::curves_are_aligned(forward, true));
  REQUIRE(tf::cpp::test::curves_are_aligned(reverse, true));
  REQUIRE(tf::cpp::test::curves_are_aligned(self, true));
  const auto forward_signature = tf::cpp::test::canonicalize_curves_geometry(
      forward, tf::cpp::test::orientation_mode::ignore, true);
  CHECK(forward_signature ==
        tf::cpp::test::canonicalize_curves_geometry(
            reverse, tf::cpp::test::orientation_mode::ignore, true));
  CHECK(forward_signature ==
        tf::cpp::test::canonicalize_curves_geometry(
            self, tf::cpp::test::orientation_mode::ignore, true));

  const auto repeated =
      tf::cpp::intersection_curves(horizontal.mesh(), vertical.mesh(), config);
  CHECK(forward_signature ==
        tf::cpp::test::canonicalize_curves_geometry(
            repeated, tf::cpp::test::orientation_mode::ignore, true));

  // the entries read their operands and write nothing back to them
  CHECK(tf::cpp::test::same_array(
      tf::cpp::test::parity_faces(horizontal.polygons), horizontal_faces));
  CHECK(tf::cpp::test::same_array(
      tf::cpp::test::parity_points(horizontal.polygons), horizontal_points));
  CHECK(tf::cpp::test::same_array(
      tf::cpp::test::parity_faces(vertical.polygons), vertical_faces));
  CHECK(tf::cpp::test::same_array(
      tf::cpp::test::parity_points(vertical.polygons), vertical_points));
  CHECK(tf::cpp::test::same_array(
      tf::cpp::test::parity_faces(combined.polygons), combined_faces));
  CHECK(tf::cpp::test::same_array(
      tf::cpp::test::parity_points(combined.polygons), combined_points));
}

TEMPLATE_TEST_CASE(
    "Python-parity intersection curves honor stored transforms and validation",
    "[cpp][intersect][python-parity][transform][validation]", float, double) {
  const auto baseline_horizontal = horizontal_triangle<TestType>();
  const auto baseline_vertical = vertical_triangle<TestType>();
  const auto baseline = tf::cpp::intersection_curves(baseline_horizontal.mesh(),
                                                     baseline_vertical.mesh());
  REQUIRE(tf::cpp::test::curves_are_aligned(baseline, true));

  auto transformed_horizontal = horizontal_triangle<TestType>(TestType{10});
  transformed_horizontal.place(parity_x_translation<TestType>(TestType{-10}));
  const auto transformed_vertical = vertical_triangle<TestType>();
  const auto local_faces =
      tf::cpp::test::parity_faces(transformed_horizontal.polygons).deep_copy();
  const auto local_points =
      tf::cpp::test::parity_points(transformed_horizontal.polygons).deep_copy();
  const auto transformed = tf::cpp::intersection_curves(
      transformed_horizontal.mesh(), transformed_vertical.mesh());
  REQUIRE(tf::cpp::test::curves_are_aligned(transformed, true));
  CHECK(tf::cpp::test::canonicalize_curves_geometry(
            baseline, tf::cpp::test::orientation_mode::ignore, true) ==
        tf::cpp::test::canonicalize_curves_geometry(
            transformed, tf::cpp::test::orientation_mode::ignore, true));
  // the placement moved the reading, not the coordinates the caller holds
  CHECK(tf::cpp::test::same_array(
      tf::cpp::test::parity_faces(transformed_horizontal.polygons),
      local_faces));
  CHECK(tf::cpp::test::same_array(
      tf::cpp::test::parity_points(transformed_horizontal.polygons),
      local_points));
  CHECK(transformed_horizontal.placement.placed);

  // faces that name a point the geometry does not have are refused at the one
  // door the reading passes
  const parity_owned<TestType> malformed{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, TestType>(
          {0, 1, 2}, {TestType{0}, TestType{0}, TestType{0}, TestType{1},
                      TestType{0}, TestType{0}})};
  CHECK_THROWS_AS(tf::cpp::intersection_curves(malformed.mesh(),
                                               transformed_vertical.mesh()),
                  std::out_of_range);

  const auto empty = tf::cpp::test::empty_mesh<TestType>();
  CHECK_THROWS_AS(
      tf::cpp::intersection_curves(empty.mesh(), transformed_vertical.mesh(),
                                   {tf::intersect_mode::primitives,
                                    std::numeric_limits<double>::infinity()}),
      std::invalid_argument);
}

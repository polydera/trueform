/**
 * @file test_determinant_of.cpp
 * @brief The determinant of a transformation's linear part, and the sign of
 *        its rotation, through every overload of the family
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/determinant_of.hpp>
#include <trueform/core/determinant_of_rotation.hpp>
#include <trueform/core/frame.hpp>
#include <trueform/core/frame_like.hpp>
#include <trueform/core/transformation.hpp>
#include <trueform/core/transformation_like.hpp>

namespace {

// the identity overloads resolve by partial ordering, as constants
static_assert(tf::determinant_of(tf::identity_transformation<double, 3>{}) ==
              1.0);
static_assert(tf::determinant_of(tf::identity_transformation<float, 2>{}) ==
              1.0f);
static_assert(tf::determinant_of(tf::identity_frame<double, 3>{}) == 1.0);
static_assert(tf::determinant_of_rotation(
                  tf::identity_transformation<double, 3>{}) == 1.0);
static_assert(tf::determinant_of_rotation(tf::identity_frame<double, 3>{}) ==
              1.0);

auto determinant_diag3(double a, double b, double c)
    -> tf::transformation<double, 3> {
  tf::transformation<double, 3> t;
  t.fill({a, 0.0, 0.0, 1.0, 0.0, b, 0.0, 2.0, 0.0, 0.0, c, 3.0});
  return t;
}

auto determinant_diag2(double a, double b) -> tf::transformation<double, 2> {
  tf::transformation<double, 2> t;
  t.fill({a, 0.0, 1.0, 0.0, b, 2.0});
  return t;
}

} // namespace

TEST_CASE("determinant_of: the linear part of a transformation and a frame",
          "[core][determinant]") {
  SECTION("3D: a non-uniform positive scale is the product of its scales") {
    const auto t = determinant_diag3(2, 3, 4);
    REQUIRE(tf::determinant_of(t) == 24.0);
    REQUIRE(tf::determinant_of(tf::make_frame(t)) == 24.0);
    REQUIRE(tf::determinant_of_rotation(t) == 1.0);
    REQUIRE(tf::determinant_of_rotation(tf::make_frame(t)) == 1.0);
  }
  SECTION("3D: a reflection is negative") {
    const auto t = determinant_diag3(2, 3, -4);
    REQUIRE(tf::determinant_of(t) == -24.0);
    REQUIRE(tf::determinant_of(tf::make_frame(t)) == -24.0);
    REQUIRE(tf::determinant_of_rotation(t) == -1.0);
    REQUIRE(tf::determinant_of_rotation(tf::make_frame(t)) == -1.0);
  }
  SECTION("3D: a shear keeps the volume") {
    tf::transformation<double, 3> t;
    t.fill({1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0});
    REQUIRE(tf::determinant_of(t) == 1.0);
    REQUIRE(tf::determinant_of_rotation(t) == 1.0);
  }
  SECTION("2D: a non-uniform positive scale and a reflection") {
    REQUIRE(tf::determinant_of(determinant_diag2(2, 5)) == 10.0);
    REQUIRE(tf::determinant_of_rotation(determinant_diag2(2, 5)) == 1.0);
    REQUIRE(tf::determinant_of(determinant_diag2(-2, 5)) == -10.0);
    REQUIRE(tf::determinant_of_rotation(determinant_diag2(-2, 5)) == -1.0);
    REQUIRE(tf::determinant_of(tf::make_frame(determinant_diag2(-2, 5))) ==
            -10.0);
  }
  SECTION("the identity, untagged and as a frame") {
    REQUIRE(tf::determinant_of(tf::identity_transformation<double, 3>{}) ==
            1.0);
    REQUIRE(tf::determinant_of(tf::identity_frame<float, 3>{}) == 1.0f);
    REQUIRE(tf::determinant_of_rotation(tf::identity_frame<float, 2>{}) ==
            1.0f);
  }
}

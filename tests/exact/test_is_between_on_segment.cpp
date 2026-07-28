/**
 * @file test_is_between_on_segment.cpp
 * @brief Tests for tf::exact::is_between_on_segment
 *
 * The single producer of betweenness: strictly inside means the exact
 * projection lies strictly between zero and the squared length. Pins the
 * strictness at both endpoints, one-lattice-unit interiority at the edge of
 * the coordinate range, the answer being a projection rather than a
 * distance, and the degenerate segment having no inside.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/is_between_on_segment.hpp>
#include <trueform/exact/vertex.hpp>

#include <limits>

TEMPLATE_TEST_CASE("interior points are between, endpoints are not",
                   "[exact][is_between_on_segment]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  const tf::exact::pt2<Int> a{Int(-1000), Int(0)};
  const tf::exact::pt2<Int> b{Int(1000), Int(0)};

  CHECK(tf::exact::is_between_on_segment<Int>(a, b, {Int(0), Int(0)}));
  CHECK(tf::exact::is_between_on_segment<Int>(a, b, {Int(-999), Int(0)}));
  CHECK(tf::exact::is_between_on_segment<Int>(a, b, {Int(999), Int(0)}));

  CHECK_FALSE(tf::exact::is_between_on_segment<Int>(a, b, a));
  CHECK_FALSE(tf::exact::is_between_on_segment<Int>(a, b, b));
  CHECK_FALSE(
      tf::exact::is_between_on_segment<Int>(a, b, {Int(-1001), Int(0)}));
  CHECK_FALSE(
      tf::exact::is_between_on_segment<Int>(a, b, {Int(1001), Int(0)}));
}

TEMPLATE_TEST_CASE("one lattice unit inside the coordinate range is exact",
                   "[exact][is_between_on_segment]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  const Int maximum = std::numeric_limits<Int>::max();
  const tf::exact::pt2<Int> a{Int(-maximum), Int(-maximum)};
  const tf::exact::pt2<Int> b{maximum, maximum};

  CHECK(tf::exact::is_between_on_segment<Int>(
      a, b, {Int(maximum - Int(1)), Int(maximum - Int(1))}));
  CHECK(tf::exact::is_between_on_segment<Int>(
      a, b, {Int(-maximum + Int(1)), Int(-maximum + Int(1))}));
  CHECK_FALSE(tf::exact::is_between_on_segment<Int>(a, b, a));
  CHECK_FALSE(tf::exact::is_between_on_segment<Int>(a, b, b));
}

TEMPLATE_TEST_CASE("the answer is a projection, not a distance",
                   "[exact][is_between_on_segment]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  const tf::exact::pt2<Int> a{Int(0), Int(0)};
  const tf::exact::pt2<Int> b{Int(1000), Int(0)};

  CHECK(tf::exact::is_between_on_segment<Int>(a, b, {Int(500), Int(700)}));
  CHECK_FALSE(
      tf::exact::is_between_on_segment<Int>(a, b, {Int(-1), Int(700)}));
  CHECK_FALSE(
      tf::exact::is_between_on_segment<Int>(a, b, {Int(1001), Int(700)}));
}

TEMPLATE_TEST_CASE("a degenerate segment has no inside",
                   "[exact][is_between_on_segment]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  const tf::exact::pt2<Int> a{Int(7), Int(7)};

  CHECK_FALSE(tf::exact::is_between_on_segment<Int>(a, a, a));
  CHECK_FALSE(
      tf::exact::is_between_on_segment<Int>(a, a, {Int(8), Int(7)}));
}

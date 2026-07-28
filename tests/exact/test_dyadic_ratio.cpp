/**
 * @file test_dyadic_ratio.cpp
 * @brief Tests for tf::exact::dyadic_ratio
 *
 * The inverse of tf::exact::dyadic_blend: a position along an edge turned
 * back into a transportable parameter. Pins the clamping contract, the
 * nearest rounding, the round trip through a blend, and the headroom of
 * the numerator at the largest three-dimensional squared edge length the
 * lattice admits -- the length param_bits is budgeted for.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/dyadic_blend.hpp>
#include <trueform/exact/dyadic_ratio.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/vertex.hpp>

#include <limits>

namespace {

template <typename Int>
auto maximum_parameter() -> typename tf::exact::meta<Int>::param_type {
  return typename tf::exact::meta<Int>::param_type(1)
         << tf::exact::meta<Int>::param_bits;
}

/// The largest squared length a three-dimensional edge on this lattice can
/// have: every coordinate spanning the whole type.
template <typename Int>
auto widest_squared_length() -> typename tf::exact::meta<Int>::T2 {
  using T2 = typename tf::exact::meta<Int>::T2;
  const T2 extent =
      T2(std::numeric_limits<Int>::max()) - T2(std::numeric_limits<Int>::min());
  return T2(3) * extent * extent;
}

template <typename Int>
auto blended(const tf::exact::pt3<Int> &a, const tf::exact::pt3<Int> &b,
             typename tf::exact::meta<Int>::param_type parameter)
    -> tf::exact::pt3<Int> {
  tf::exact::pt3<Int> point;
  for (int coordinate = 0; coordinate < 3; ++coordinate)
    point[coordinate] =
        tf::exact::dyadic_blend(a[coordinate], b[coordinate], parameter);
  return point;
}

template <typename Int>
auto parameter_of(const tf::exact::pt3<Int> &a, const tf::exact::pt3<Int> &b,
                  const tf::exact::pt3<Int> &point) ->
    typename tf::exact::meta<Int>::param_type {
  using T2 = typename tf::exact::meta<Int>::T2;
  T2 numerator(0);
  T2 denominator(0);
  for (int coordinate = 0; coordinate < 3; ++coordinate) {
    const T2 delta = T2(b[coordinate]) - T2(a[coordinate]);
    numerator =
        numerator + (T2(point[coordinate]) - T2(a[coordinate])) * delta;
    denominator = denominator + delta * delta;
  }
  return tf::exact::dyadic_ratio<Int>(numerator, denominator);
}

} // namespace

TEMPLATE_TEST_CASE("dyadic_ratio clamps to the endpoints",
                   "[exact][dyadic_ratio]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto maximum = maximum_parameter<Int>();
  const T2 denominator(1000);

  CHECK(tf::exact::dyadic_ratio<Int>(T2(0), denominator) == param_t(0));
  CHECK(tf::exact::dyadic_ratio<Int>(T2(-1), denominator) == param_t(0));
  CHECK(tf::exact::dyadic_ratio<Int>(T2(-denominator), denominator) ==
        param_t(0));
  CHECK(tf::exact::dyadic_ratio<Int>(denominator, T2(0)) == param_t(0));
  CHECK(tf::exact::dyadic_ratio<Int>(denominator, T2(-1000)) == param_t(0));
  CHECK(tf::exact::dyadic_ratio<Int>(denominator, denominator) == maximum);
  CHECK(tf::exact::dyadic_ratio<Int>(denominator + T2(1), denominator) ==
        maximum);
}

TEMPLATE_TEST_CASE("dyadic_ratio is exact on dyadic fractions",
                   "[exact][dyadic_ratio]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto maximum = maximum_parameter<Int>();

  CHECK(tf::exact::dyadic_ratio<Int>(T2(1), T2(2)) == maximum / param_t(2));
  CHECK(tf::exact::dyadic_ratio<Int>(T2(1), T2(4)) == maximum / param_t(4));
  CHECK(tf::exact::dyadic_ratio<Int>(T2(3), T2(4)) ==
        (maximum / param_t(4)) * param_t(3));
  CHECK(tf::exact::dyadic_ratio<Int>(T2(1), T2(8)) == maximum / param_t(8));
  CHECK(tf::exact::dyadic_ratio<Int>(T2(7), T2(8)) ==
        (maximum / param_t(8)) * param_t(7));
}

TEMPLATE_TEST_CASE("dyadic_ratio rounds a sub-quantum position onto an endpoint",
                   "[exact][dyadic_ratio]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto maximum = maximum_parameter<Int>();
  const T2 scale = T2(maximum);

  // Denominators of 2^(param_bits + 1) put the numerator exactly half a
  // quantum from an end.
  CHECK(tf::exact::dyadic_ratio<Int>(T2(1), T2(2) * scale) == param_t(1));
  CHECK(tf::exact::dyadic_ratio<Int>(T2(1), T2(2) * scale + T2(2)) ==
        param_t(0));
  CHECK(tf::exact::dyadic_ratio<Int>(T2(2) * scale - T2(1), T2(2) * scale) ==
        maximum);
}

TEMPLATE_TEST_CASE("dyadic_ratio recovers the parameter of an exact blend",
                   "[exact][dyadic_ratio]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto maximum = maximum_parameter<Int>();
  // Coordinates that are multiples of 32 make every blend at a multiple of
  // maximum / 32 land on the lattice with no rounding at all, so the
  // recovered parameter is the one the point was placed from.
  const tf::exact::pt3<Int> a{Int(-1048576), Int(524288), Int(-32)};
  const tf::exact::pt3<Int> b{Int(2097152), Int(-1048576), Int(65536)};

  for (int step = 0; step <= 32; ++step) {
    const param_t parameter = (maximum / param_t(32)) * param_t(step);
    const auto point = blended<Int>(a, b, parameter);
    CHECK(parameter_of<Int>(a, b, point) == parameter);
    const auto again = blended<Int>(a, b, parameter_of<Int>(a, b, point));
    CHECK(again[0] == point[0]);
    CHECK(again[1] == point[1]);
    CHECK(again[2] == point[2]);
  }
}

TEMPLATE_TEST_CASE("dyadic_ratio round trip holds the endpoints and the chord",
                   "[exact][dyadic_ratio]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto maximum = maximum_parameter<Int>();
  const tf::exact::pt3<Int> a{Int(-7), Int(13), Int(-1000003)};
  const tf::exact::pt3<Int> b{Int(73741823), Int(999999), Int(771)};

  CHECK(parameter_of<Int>(a, b, a) == param_t(0));
  CHECK(parameter_of<Int>(a, b, b) == maximum);

  // A coordinate rounded onto the lattice no longer projects exactly onto
  // its own parameter, so re-blending it can move it by one lattice unit.
  // That bound is why the parameter, not the coordinate, is what carriers
  // of an edge share.
  for (int step = 0; step <= 64; ++step) {
    const param_t parameter = (maximum / param_t(64)) * param_t(step);
    const auto point = blended<Int>(a, b, parameter);
    const auto again = blended<Int>(a, b, parameter_of<Int>(a, b, point));
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      const auto drift = again[coordinate] - point[coordinate];
      CHECK(drift >= Int(-1));
      CHECK(drift <= Int(1));
    }
  }
}

TEMPLATE_TEST_CASE("dyadic_ratio has headroom at the widest squared length",
                   "[exact][dyadic_ratio]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto maximum = maximum_parameter<Int>();
  const T2 scale = T2(maximum);
  const T2 denominator = widest_squared_length<Int>();

  // One quantum of this edge is denominator / 2^param_bits lattice units,
  // far more than one, so the extremes round onto the endpoints and the
  // midpoint is exact.
  CHECK(tf::exact::dyadic_ratio<Int>(T2(1), denominator) == param_t(0));
  CHECK(tf::exact::dyadic_ratio<Int>(denominator - T2(1), denominator) ==
        maximum);
  CHECK(tf::exact::dyadic_ratio<Int>(denominator / T2(2), denominator) ==
        maximum / param_t(2));

  // Nearest-multiple property, carried in T2 itself: the product that
  // overflows first is numerator * 2^param_bits, and both sides of this
  // comparison are that size.
  for (int part = 1; part < 16; ++part) {
    const T2 numerator = (denominator / T2(16)) * T2(part) + T2(part * part);
    const T2 placed = T2(tf::exact::dyadic_ratio<Int>(numerator, denominator));
    const T2 residual = numerator * scale - placed * denominator;
    CHECK((residual < T2(0) ? -residual : residual) * T2(2) <= denominator);
  }
}

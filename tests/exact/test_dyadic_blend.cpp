/**
 * @file test_dyadic_blend.cpp
 * @brief Tests for tf::exact::dyadic_blend
 *
 * The single authority that turns a parameter on an edge into a lattice
 * coordinate. Pins the exact endpoints, the round-half-away-from-zero
 * rule and its symmetry about zero, the invariance of a placement under
 * reversing the edge, and the headroom of the accumulator at the widest
 * span the lattice admits.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/dyadic_blend.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>

#include <limits>

namespace {

template <typename Int>
auto blend_maximum_parameter() -> typename tf::exact::meta<Int>::param_type {
  return typename tf::exact::meta<Int>::param_type(1)
         << tf::exact::meta<Int>::param_bits;
}

} // namespace

TEMPLATE_TEST_CASE("dyadic_blend endpoints are exact", "[exact][dyadic_blend]",
                   tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto maximum = blend_maximum_parameter<Int>();
  const Int samples[] = {Int(0),
                         Int(1),
                         Int(-1),
                         Int(7),
                         Int(-9),
                         std::numeric_limits<Int>::min(),
                         std::numeric_limits<Int>::max()};

  for (Int a : samples) {
    for (Int b : samples) {
      CHECK(tf::exact::dyadic_blend(a, b, param_t(0)) == a);
      CHECK(tf::exact::dyadic_blend(a, b, maximum) == b);
    }
  }
}

TEMPLATE_TEST_CASE("dyadic_blend rounds half away from zero",
                   "[exact][dyadic_blend]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const param_t half = blend_maximum_parameter<Int>() / param_t(2);

  // exact midpoints 0.5, -0.5, 0.5, -0.5, -1.5, 1.5
  CHECK(tf::exact::dyadic_blend(Int(0), Int(1), half) == Int(1));
  CHECK(tf::exact::dyadic_blend(Int(0), Int(-1), half) == Int(-1));
  CHECK(tf::exact::dyadic_blend(Int(-1), Int(2), half) == Int(1));
  CHECK(tf::exact::dyadic_blend(Int(1), Int(-2), half) == Int(-1));
  CHECK(tf::exact::dyadic_blend(Int(-3), Int(0), half) == Int(-2));
  CHECK(tf::exact::dyadic_blend(Int(3), Int(0), half) == Int(2));
}

TEMPLATE_TEST_CASE("dyadic_blend is symmetric about zero",
                   "[exact][dyadic_blend]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto maximum = blend_maximum_parameter<Int>();
  const Int extent = std::numeric_limits<Int>::max();
  const Int samples[] = {Int(0), Int(1), Int(-1), Int(13), Int(-97), extent,
                         Int(-extent)};
  const param_t parameters[] = {param_t(0),
                                param_t(1),
                                maximum / param_t(8),
                                maximum / param_t(3),
                                maximum / param_t(2),
                                maximum - param_t(1),
                                maximum};

  for (Int a : samples)
    for (Int b : samples)
      for (param_t parameter : parameters)
        CHECK(tf::exact::dyadic_blend(Int(-a), Int(-b), parameter) ==
              Int(-tf::exact::dyadic_blend(a, b, parameter)));

  // A span from -extent to extent must not lean toward either end: its
  // midpoint is the origin and mirrored parameters give mirrored points.
  CHECK(tf::exact::dyadic_blend(Int(-extent), extent, maximum / param_t(2)) ==
        Int(0));
  for (param_t parameter : parameters)
    CHECK(tf::exact::dyadic_blend(Int(-extent), extent, parameter) ==
          Int(-tf::exact::dyadic_blend(Int(-extent), extent,
                                       maximum - parameter)));
}

TEMPLATE_TEST_CASE("dyadic_blend places a split from either end of the edge",
                   "[exact][dyadic_blend]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto maximum = blend_maximum_parameter<Int>();
  const Int a = Int(-123457);
  const Int b = std::numeric_limits<Int>::max();
  const param_t parameters[] = {param_t(0),
                                param_t(1),
                                maximum / param_t(7),
                                maximum / param_t(2),
                                (maximum / param_t(6)) * param_t(5),
                                maximum - param_t(1),
                                maximum};

  // Two carriers holding the same edge in opposite orientations must place
  // the split on the same lattice point, or the seam between them opens.
  for (param_t parameter : parameters)
    CHECK(tf::exact::dyadic_blend(a, b, parameter) ==
          tf::exact::dyadic_blend(b, a, maximum - parameter));
}

TEMPLATE_TEST_CASE("dyadic_blend has headroom at the widest lattice span",
                   "[exact][dyadic_blend]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto maximum = blend_maximum_parameter<Int>();
  const param_t quarter = maximum / param_t(4);
  const Int lattice_min = std::numeric_limits<Int>::min();
  const Int lattice_max = std::numeric_limits<Int>::max();
  const Int half_max = Int(lattice_max >> 1);

  // The widest span the type admits, blended at the accumulator's worst
  // weights. Exact values: (min + max) / 2 = -1/2 -> -1;
  // (3 min + max) / 4 = -(2^coordinate_bits + 1) / 4 -> -(half_max + 1);
  // (min + 3 max) / 4 = (2^coordinate_bits - 3) / 4 -> half_max.
  CHECK(tf::exact::dyadic_blend(lattice_min, lattice_max,
                                maximum / param_t(2)) == Int(-1));
  CHECK(tf::exact::dyadic_blend(lattice_min, lattice_max, quarter) ==
        Int(-(half_max + Int(1))));
  CHECK(tf::exact::dyadic_blend(lattice_min, lattice_max,
                                param_t(3) * quarter) == half_max);

  // The same span made symmetric: the quarter points are +-(max + 1) / 2.
  CHECK(tf::exact::dyadic_blend(Int(-lattice_max), lattice_max, quarter) ==
        Int(-(half_max + Int(1))));
  CHECK(tf::exact::dyadic_blend(Int(-lattice_max), lattice_max,
                                param_t(3) * quarter) ==
        Int(half_max + Int(1)));
}

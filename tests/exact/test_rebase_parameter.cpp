/**
 * @file test_rebase_parameter.cpp
 * @brief Tests for tf::exact::rebase_parameter
 *
 * A position measured on a piece of an edge, restated on the whole edge.
 * Pins the identity when the piece is the parent, the reversal when the
 * parent runs the other way, the exactness of the piece's own endpoints,
 * the round-half-away-from-zero rule shared with tf::exact::dyadic_blend,
 * and the propagation of the not-a-parameter sentinel.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/rebase_parameter.hpp>

namespace {

template <typename Int>
auto maximum_parameter() -> typename tf::exact::meta<Int>::param_type {
  return typename tf::exact::meta<Int>::param_type(1)
         << tf::exact::meta<Int>::param_bits;
}

} // namespace

TEMPLATE_TEST_CASE("rebase_parameter is the identity on the whole parent",
                   "[exact][rebase_parameter]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto maximum = maximum_parameter<Int>();
  const param_t locals[] = {param_t(0),
                            param_t(1),
                            maximum / param_t(3),
                            maximum / param_t(2),
                            (maximum / param_t(4)) * param_t(3),
                            maximum - param_t(1),
                            maximum};

  for (param_t local : locals)
    CHECK(tf::exact::rebase_parameter<Int>(local, param_t(0), maximum,
                                           T2(maximum)) == local);
}

TEMPLATE_TEST_CASE("rebase_parameter reverses a backwards parent",
                   "[exact][rebase_parameter]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto maximum = maximum_parameter<Int>();
  const param_t locals[] = {param_t(0),
                            param_t(1),
                            maximum / param_t(3),
                            maximum / param_t(2),
                            (maximum / param_t(4)) * param_t(3),
                            maximum - param_t(1),
                            maximum};

  for (param_t local : locals)
    CHECK(tf::exact::rebase_parameter<Int>(local, maximum, param_t(0),
                                           T2(maximum)) == maximum - local);
}

TEMPLATE_TEST_CASE("rebase_parameter places a piece's own ends exactly",
                   "[exact][rebase_parameter]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto maximum = maximum_parameter<Int>();
  const param_t eighth = maximum / param_t(8);
  const param_t local_maximum = param_t(1) << 20;
  const T2 denominator = T2(local_maximum);

  // A split at the end of a piece is the piece's end on the parent, in
  // either orientation: no drift is allowed to accumulate there.
  CHECK(tf::exact::rebase_parameter<Int>(param_t(0), eighth * param_t(3),
                                         eighth * param_t(7),
                                         denominator) == eighth * param_t(3));
  CHECK(tf::exact::rebase_parameter<Int>(local_maximum, eighth * param_t(3),
                                         eighth * param_t(7),
                                         denominator) == eighth * param_t(7));
  CHECK(tf::exact::rebase_parameter<Int>(param_t(0), eighth * param_t(7),
                                         eighth * param_t(3),
                                         denominator) == eighth * param_t(7));
  CHECK(tf::exact::rebase_parameter<Int>(local_maximum, eighth * param_t(7),
                                         eighth * param_t(3),
                                         denominator) == eighth * param_t(3));

  // The middle of the piece is the middle of its span on the parent,
  // whichever way the parent runs.
  CHECK(tf::exact::rebase_parameter<Int>(local_maximum / param_t(2),
                                         eighth * param_t(3),
                                         eighth * param_t(7),
                                         denominator) == eighth * param_t(5));
  CHECK(tf::exact::rebase_parameter<Int>(local_maximum / param_t(2),
                                         eighth * param_t(7),
                                         eighth * param_t(3),
                                         denominator) == eighth * param_t(5));
}

TEMPLATE_TEST_CASE("rebase_parameter rounds half away from zero",
                   "[exact][rebase_parameter]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto maximum = maximum_parameter<Int>();
  const param_t half = maximum / param_t(2);

  // One unit on a piece covering the parent's first half is half a unit of
  // the parent: away from zero in both directions of travel.
  CHECK(tf::exact::rebase_parameter<Int>(param_t(1), param_t(0), half,
                                         T2(maximum)) == param_t(1));
  CHECK(tf::exact::rebase_parameter<Int>(param_t(1), half, param_t(0),
                                         T2(maximum)) == half - param_t(1));
}

TEMPLATE_TEST_CASE("rebase_parameter propagates the not-a-parameter sentinel",
                   "[exact][rebase_parameter]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto maximum = maximum_parameter<Int>();

  CHECK(tf::exact::rebase_parameter<Int>(param_t(-1), param_t(0), maximum,
                                         T2(maximum)) == param_t(-1));
  CHECK(tf::exact::rebase_parameter<Int>(param_t(-5), maximum, param_t(0),
                                         T2(maximum)) == param_t(-1));
  CHECK(tf::exact::rebase_parameter<Int>(maximum / param_t(2), param_t(0),
                                         maximum, T2(0)) == param_t(-1));
  CHECK(tf::exact::rebase_parameter<Int>(maximum / param_t(2), param_t(0),
                                         maximum, T2(-1)) == param_t(-1));
  CHECK(tf::exact::rebase_parameter<Int>(param_t(0), param_t(0), maximum,
                                         T2(0)) == param_t(-1));
}

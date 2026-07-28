/**
 * @file test_snap_parameter_to_edge.cpp
 * @brief Tests for tf::exact::edge_parameter_step_bits and
 *        tf::exact::snap_parameter_to_edge
 *
 * The quantisation that decides how far apart two positions on one edge
 * must be to be two splits. Pins the step derived from the edge's own
 * extent, the collapse of an edge with no interior at the source type's
 * resolution, the deliberate rounding of a near-endpoint position onto the
 * endpoint, and the lattice size of one step.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/dyadic_blend.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/snap_parameter_to_edge.hpp>
#include <trueform/exact/vertex.hpp>

#include <limits>

namespace {

template <typename Int>
auto maximum_parameter() -> typename tf::exact::meta<Int>::param_type {
  return typename tf::exact::meta<Int>::param_type(1)
         << tf::exact::meta<Int>::param_bits;
}

/// Bits of the source real type's significand: the number of steps the
/// longest edge of the lattice is divided into.
template <typename Int> auto significand_bits() -> int {
  return std::numeric_limits<typename tf::exact::meta<Int>::real_type>::digits;
}

/// Lattice units one parameter step covers on an axis edge of 2^bits.
template <typename Int> auto one_step_in_lattice_units(int bits) -> Int {
  using param_t = typename tf::exact::meta<Int>::param_type;
  const Int extent = Int(Int(1) << bits);
  const tf::exact::pt3<Int> a{Int(0), Int(0), Int(0)};
  const tf::exact::pt3<Int> b{extent, Int(0), Int(0)};
  const param_t step =
      param_t(1) << tf::exact::edge_parameter_step_bits<Int>(a, b);
  return tf::exact::dyadic_blend(a[0], b[0], step);
}

} // namespace

TEMPLATE_TEST_CASE("edge_parameter_step_bits follows the edge's own extent",
                   "[exact][snap_parameter_to_edge]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  constexpr int param_bits = tf::exact::meta<Int>::param_bits;
  constexpr int ulp_bits = tf::exact::meta<Int>::conversion_ulp_bits;
  constexpr int coordinate_bits = tf::exact::meta<Int>::coordinate_bits;
  const tf::exact::pt3<Int> origin{Int(0), Int(0), Int(0)};

  CHECK(tf::exact::edge_parameter_step_bits<Int>(
            tf::exact::pt3<Int>{Int(5), Int(5), Int(5)},
            tf::exact::pt3<Int>{Int(5), Int(5), Int(5)}) == param_bits);
  CHECK(tf::exact::edge_parameter_step_bits<Int>(
            origin, tf::exact::pt3<Int>{Int((Int(1) << ulp_bits) - Int(1)),
                                        Int(0), Int(0)}) == param_bits);
  CHECK(tf::exact::edge_parameter_step_bits<Int>(
            origin, tf::exact::pt3<Int>{Int(Int(1) << ulp_bits), Int(0),
                                        Int(0)}) == param_bits - 1);
  CHECK(tf::exact::edge_parameter_step_bits<Int>(
            origin,
            tf::exact::pt3<Int>{Int(Int(1) << (coordinate_bits - 1)), Int(0),
                                Int(0)}) == param_bits - significand_bits<Int>());

  // The extent is the largest coordinate difference in magnitude, so the
  // axis it falls on and the direction of travel do not change the step.
  const tf::exact::pt3<Int> a{Int(5), Int(-100), Int(3)};
  const tf::exact::pt3<Int> b{Int(5), Int(27), Int(3)};
  CHECK(tf::exact::edge_parameter_step_bits<Int>(a, b) ==
        tf::exact::edge_parameter_step_bits<Int>(b, a));
  CHECK(tf::exact::edge_parameter_step_bits<Int>(a, b) ==
        tf::exact::edge_parameter_step_bits<Int>(
            origin, tf::exact::pt3<Int>{Int(127), Int(0), Int(0)}));
}

TEMPLATE_TEST_CASE("snap_parameter_to_edge collapses an edge with no interior",
                   "[exact][snap_parameter_to_edge]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  const auto maximum = maximum_parameter<Int>();
  const tf::exact::pt3<Int> a{Int(-11), Int(4), Int(9)};
  const tf::exact::pt3<Int> b{Int(-10), Int(4), Int(9)};

  // One step covers the whole edge, so no position on it survives as a
  // split: each one snaps to the end it is nearer.
  CHECK(tf::exact::snap_parameter_to_edge<Int>(param_t(1), a, b) ==
        param_t(0));
  CHECK(tf::exact::snap_parameter_to_edge<Int>(maximum / param_t(4), a, b) ==
        param_t(0));
  CHECK(tf::exact::snap_parameter_to_edge<Int>(maximum / param_t(2) -
                                                   param_t(1),
                                               a, b) == param_t(0));
  CHECK(tf::exact::snap_parameter_to_edge<Int>(maximum / param_t(2), a, b) ==
        maximum);
  CHECK(tf::exact::snap_parameter_to_edge<Int>(maximum - param_t(1), a, b) ==
        maximum);
  CHECK(tf::exact::snap_parameter_to_edge<Int>(maximum, a, b) == maximum);
}

TEMPLATE_TEST_CASE("snap_parameter_to_edge rounds onto the nearest step",
                   "[exact][snap_parameter_to_edge]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  constexpr int coordinate_bits = tf::exact::meta<Int>::coordinate_bits;
  const auto maximum = maximum_parameter<Int>();
  const tf::exact::pt3<Int> a{Int(0), Int(0), Int(0)};
  const tf::exact::pt3<Int> b{Int(Int(1) << (coordinate_bits - 1)), Int(0),
                              Int(0)};
  const param_t step =
      param_t(1) << tf::exact::edge_parameter_step_bits<Int>(a, b);
  const param_t half_step = step / param_t(2);

  CHECK(tf::exact::snap_parameter_to_edge<Int>(param_t(0), a, b) ==
        param_t(0));
  CHECK(tf::exact::snap_parameter_to_edge<Int>(step, a, b) == step);
  CHECK(tf::exact::snap_parameter_to_edge<Int>(step * param_t(9), a, b) ==
        step * param_t(9));
  CHECK(tf::exact::snap_parameter_to_edge<Int>(half_step - param_t(1), a, b) ==
        param_t(0));
  CHECK(tf::exact::snap_parameter_to_edge<Int>(half_step, a, b) == step);
  CHECK(tf::exact::snap_parameter_to_edge<Int>(step + half_step - param_t(1),
                                               a, b) == step);

  // The rounding is deliberate at the far end too: a position within half a
  // step of the endpoint becomes the endpoint rather than a split beside it.
  CHECK(tf::exact::snap_parameter_to_edge<Int>(maximum - param_t(1), a, b) ==
        maximum);
  CHECK(tf::exact::snap_parameter_to_edge<Int>(maximum - half_step, a, b) ==
        maximum);
  CHECK(tf::exact::snap_parameter_to_edge<Int>(maximum - half_step -
                                                   param_t(1),
                                               a, b) == maximum - step);

  const param_t parameters[] = {param_t(0),
                                param_t(1),
                                half_step,
                                step + param_t(3),
                                maximum / param_t(3),
                                maximum / param_t(2) + param_t(7),
                                maximum - param_t(1),
                                maximum};
  for (param_t parameter : parameters) {
    const auto snapped = tf::exact::snap_parameter_to_edge<Int>(parameter, a, b);
    CHECK(tf::exact::snap_parameter_to_edge<Int>(snapped, a, b) == snapped);
    CHECK(snapped >= param_t(0));
    CHECK(snapped <= maximum);
  }
}

TEMPLATE_TEST_CASE("snap_parameter_to_edge steps by the source type's ulp",
                   "[exact][snap_parameter_to_edge]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  constexpr int coordinate_bits = tf::exact::meta<Int>::coordinate_bits;
  const Int extent = Int(Int(1) << (coordinate_bits - 1));
  const tf::exact::pt3<Int> a{Int(0), Int(0), Int(0)};
  const tf::exact::pt3<Int> b{extent, Int(0), Int(0)};
  const param_t step =
      param_t(1) << tf::exact::edge_parameter_step_bits<Int>(a, b);

  // One step along the edge is the extent divided by the significand of the
  // real type the lattice was converted from.
  CHECK(tf::exact::dyadic_blend(a[0], b[0], step) ==
        Int(extent >> significand_bits<Int>()));
}

TEMPLATE_TEST_CASE("snap_parameter_to_edge makes two positions inside one step "
                   "one parameter",
                   "[exact][snap_parameter_to_edge]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  constexpr int coordinate_bits = tf::exact::meta<Int>::coordinate_bits;
  const tf::exact::pt3<Int> a{Int(0), Int(0), Int(0)};
  const tf::exact::pt3<Int> b{Int(Int(1) << (coordinate_bits - 1)), Int(0),
                              Int(0)};
  const param_t step =
      param_t(1) << tf::exact::edge_parameter_step_bits<Int>(a, b);
  const param_t half_step = step / param_t(2);
  const param_t base = step * param_t(100);

  // Distinct raw parameters the dedup downstream would keep apart, all
  // stated on one grid point once they are snapped.
  const param_t offsets[] = {param_t(1) - half_step,
                             -half_step / param_t(2),
                             param_t(-1),
                             param_t(0),
                             param_t(1),
                             half_step / param_t(2),
                             half_step - param_t(1)};
  for (param_t offset : offsets)
    CHECK(tf::exact::snap_parameter_to_edge<Int>(base + offset, a, b) == base);

  // Two positions further apart than a step are never the same position.
  const param_t spread[] = {base, base + step + param_t(1),
                            base + param_t(3) * step - param_t(2),
                            base + param_t(11) * step + half_step};
  for (param_t left : spread)
    for (param_t right : spread) {
      const auto snapped_left =
          tf::exact::snap_parameter_to_edge<Int>(left, a, b);
      const auto snapped_right =
          tf::exact::snap_parameter_to_edge<Int>(right, a, b);
      const param_t distance =
          left < right ? right - left : left - right;
      if (distance > step)
        CHECK(snapped_left != snapped_right);
    }
}

TEMPLATE_TEST_CASE("snap_parameter_to_edge lands an interior position on an "
                   "endpoint",
                   "[exact][snap_parameter_to_edge]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using param_t = typename tf::exact::meta<Int>::param_type;
  constexpr int coordinate_bits = tf::exact::meta<Int>::coordinate_bits;
  const auto maximum = maximum_parameter<Int>();
  const tf::exact::pt3<Int> a{Int(-3), Int(7), Int(0)};
  const tf::exact::pt3<Int> b{Int(-3), Int(7),
                              Int(Int(1) << (coordinate_bits / 2))};
  const param_t step =
      param_t(1) << tf::exact::edge_parameter_step_bits<Int>(a, b);
  const param_t half_step = step / param_t(2);

  // Strictly interior parameters, and yet the position they name is an end
  // of the edge: the producer has to drop them rather than materialize a
  // second identity onto a vertex that already exists.
  const param_t near_low[] = {param_t(1), half_step / param_t(2),
                              half_step - param_t(1)};
  for (param_t parameter : near_low) {
    CHECK(parameter > param_t(0));
    CHECK(tf::exact::snap_parameter_to_edge<Int>(parameter, a, b) ==
          param_t(0));
  }
  const param_t near_high[] = {maximum - param_t(1),
                               maximum - half_step / param_t(2),
                               maximum - half_step};
  for (param_t parameter : near_high) {
    CHECK(parameter < maximum);
    CHECK(tf::exact::snap_parameter_to_edge<Int>(parameter, a, b) == maximum);
  }

  // Just outside the endpoint's reach the position survives as its own.
  CHECK(tf::exact::snap_parameter_to_edge<Int>(half_step, a, b) == step);
  CHECK(tf::exact::snap_parameter_to_edge<Int>(maximum - half_step -
                                                   param_t(1),
                                               a, b) == maximum - step);
}

TEMPLATE_TEST_CASE("edge_parameter_step_bits is one lattice distance on every "
                   "edge",
                   "[exact][snap_parameter_to_edge]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  constexpr int ulp_bits = tf::exact::meta<Int>::conversion_ulp_bits;
  constexpr int coordinate_bits = tf::exact::meta<Int>::coordinate_bits;
  const Int expected = Int(Int(1) << (ulp_bits - 1));

  // Edges spanning four orders of magnitude of the lattice, and one step
  // along each of them covers the same lattice distance.
  const int extents[] = {ulp_bits + 2, coordinate_bits / 3,
                         coordinate_bits / 2, coordinate_bits - 1};
  for (int bits : extents)
    CHECK(one_step_in_lattice_units<Int>(bits) == expected);
}

TEST_CASE("edge_parameter_step_bits follows the lattice it is asked about",
          "[exact][snap_parameter_to_edge]") {
  const auto narrow = one_step_in_lattice_units<tf::exact::int32>(
      tf::exact::meta<tf::exact::int32>::coordinate_bits / 2);
  const auto wide = one_step_in_lattice_units<tf::exact::int64>(
      tf::exact::meta<tf::exact::int64>::coordinate_bits / 2);

  CHECK(narrow == tf::exact::int32(
                      tf::exact::int32(1)
                      << (tf::exact::meta<tf::exact::int32>::conversion_ulp_bits -
                          1)));
  CHECK(wide == tf::exact::int64(
                    tf::exact::int64(1)
                    << (tf::exact::meta<tf::exact::int64>::conversion_ulp_bits -
                        1)));
  CHECK(tf::exact::int64(narrow) != wide);
}

/*
 * Copyright (c) 2026 XLAB
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
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <trueform/exact/dyadic_blend.hpp>
#include <trueform/exact/dyadic_blend_scaled.hpp>
#include <trueform/exact/edge_parameter.hpp>
#include <trueform/exact/edge_point_parameter.hpp>
#include <trueform/exact/edge_projection_parameter.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/vertex.hpp>

namespace {

template <typename Int> void check_edge_lattice() {
  using P = tf::exact::pt3<Int>;
  using T2 = typename tf::exact::meta<Int>::T2;
  const P a{-16, 8, -24}, b{48, -24, 40};
  for (int n = 1; n < 32; ++n) {
    P q;
    for (int k = 0; k < 3; ++k) {
      q[k] = tf::exact::dyadic_blend<Int>(a[k], b[k], n, 5);
      REQUIRE(T2(q[k]) * T2(32) == T2(a[k]) * T2(32 - n) + T2(b[k]) * T2(n));
    }
    const auto projection =
        tf::exact::make_edge_projection_parameter<Int>(a, b, q);
    const auto incident = tf::exact::make_edge_point_parameter<Int>(a, b, q);
    const tf::exact::edge_parameter<Int> expected{T2(n), T2(32)};
    CHECK(tf::exact::compare_parameter(projection, expected) == 0);
    CHECK(tf::exact::compare_parameter(projection, incident) == 0);
  }
  const auto projected = tf::exact::make_edge_projection_parameter<Int>(
      P{0, 0, 0}, P{8, 0, 0}, P{3, 17, -6});
  CHECK(projected.num == T2(24));
  CHECK(projected.den == T2(64));
  const auto reversed = tf::exact::make_edge_projection_parameter<Int>(
      P{8, 0, 0}, P{0, 0, 0}, P{3, 17, -6});
  CHECK(tf::exact::compare_parameter(
            reversed, tf::exact::reversed_parameter(projected)) == 0);
}

} // namespace

TEST_CASE("edge lattice: dyadic placement and projection on both integer lanes",
          "[exact][edge-lattice]") {
  check_edge_lattice<tf::exact::int32>();
  check_edge_lattice<tf::exact::int64>();
}

TEST_CASE("edge lattice: scaled narrow lane keeps all coordinate bits",
          "[exact][edge-lattice]") {
  using P = tf::exact::pt3<std::int64_t>;
  const auto scale = std::int64_t(1) << 35;
  const P a{-scale, scale, 0}, b{scale, -scale, scale}, q{0, 0, scale / 2};
  const auto p =
      tf::exact::make_edge_projection_parameter<tf::exact::int32>(a, b, q);
  CHECK(p.num * 2 == p.den);
}

TEST_CASE("edge lattice: scaled wide coordinates interpolate without narrowing",
          "[exact][edge-lattice]") {
  using Int = tf::exact::int64;
  using Coord = tf::exact::meta<Int>::T1;
  const Coord origin = Coord(1) << 70, extent = Coord(1) << 20;
  for (int n = 1; n < 64; ++n) {
    const auto got = tf::exact::dyadic_blend_scaled<Int>(
        origin, origin + extent, Coord(n), 6);
    CHECK(got == origin + (extent / Coord(64)) * Coord(n));
    CHECK(tf::exact::dyadic_blend_scaled<Int>(-origin, -origin - extent,
                                              Coord(n), 6) == -got);
  }
}

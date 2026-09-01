/**
 * @file test_edge_parameter.cpp
 * @brief tf::exact::edge_parameter — exact fractions on segment carriers.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/dyadic_ratio.hpp>
#include <trueform/exact/edge_edge_parameter.hpp>
#include <trueform/exact/edge_parameter.hpp>
#include <trueform/exact/edge_plane_parameter.hpp>
#include <trueform/exact/orient3d.hpp>
#include <trueform/exact/parameter_line_sign2.hpp>
#include <trueform/exact/parameter_plane_sign.hpp>
#include <trueform/exact/projection_axes.hpp>

#include <algorithm>
#include <array>
#include <random>
#include <vector>

using tf::exact::int64;
using pt = tf::exact::pt3<int64>;

namespace {

// x = offset plane from an arbitrary triple of points on it.
auto x_plane(int64 offset, int64 s) -> std::array<pt, 3> {
  return {pt{offset, s, 2 * s + 1}, pt{offset, -3 * s - 2, s},
          pt{offset, 5 * s + 3, -s - 7}};
}

} // namespace

TEST_CASE("edge_parameter: representation invariance of the EF fraction",
          "[exact][edge_parameter]") {
  const pt p0{0, 0, 0}, p1{10, 0, 0};
  const auto pl_a = x_plane(5, 1);
  const auto pl_b = x_plane(5, 17);
  const auto t_a =
      tf::exact::make_edge_plane_parameter<int64>(pl_a[0], pl_a[1], pl_a[2],
                                                  p0, p1);
  const auto t_b =
      tf::exact::make_edge_plane_parameter<int64>(pl_b[0], pl_b[1], pl_b[2],
                                                  p0, p1);
  REQUIRE(t_a.den > decltype(t_a.den)(0));
  REQUIRE(tf::exact::compare_parameter(t_a, t_b) == 0);
}

TEST_CASE("edge_parameter: order along the carrier, ties exact",
          "[exact][edge_parameter]") {
  const pt p0{0, 0, 0}, p1{10, 0, 0};
  std::vector<int> offsets{9, 2, 5, 2, 0, 10, 7};
  std::vector<tf::exact::edge_parameter<int64>> params;
  for (const auto o : offsets) {
    const auto pl = x_plane(o, o + 3);
    params.push_back(tf::exact::make_edge_plane_parameter<int64>(
        pl[0], pl[1], pl[2], p0, p1));
  }
  for (std::size_t i = 0; i < offsets.size(); ++i)
    for (std::size_t j = 0; j < offsets.size(); ++j) {
      const int expected = (offsets[i] < offsets[j])   ? -1
                           : (offsets[i] > offsets[j]) ? 1
                                                       : 0;
      REQUIRE(tf::exact::compare_parameter(params[i], params[j]) == expected);
    }
  // endpoint plane: the exact 0
  const auto pl0 = x_plane(0, 4);
  const auto t0 = tf::exact::make_edge_plane_parameter<int64>(pl0[0], pl0[1],
                                                              pl0[2], p0, p1);
  REQUIRE(t0.num == decltype(t0.num)(0));
}

TEST_CASE("edge_parameter: the junction predicate is exactly zero on its "
          "own crossing",
          "[exact][edge_parameter]") {
  const pt p0{-7, 3, 11}, p1{13, -9, 2};
  const auto pl = x_plane(4, 6);
  const auto t = tf::exact::make_edge_plane_parameter<int64>(pl[0], pl[1],
                                                             pl[2], p0, p1);
  REQUIRE(tf::exact::parameter_plane_sign<int64>(pl[0], pl[1], pl[2], p0, p1,
                                                 t) == 0);
  // and strictly separated by any plane not through the point
  const auto pl_lo = x_plane(3, 2);
  const auto pl_hi = x_plane(5, 2);
  const int s_lo = tf::exact::parameter_plane_sign<int64>(
      pl_lo[0], pl_lo[1], pl_lo[2], p0, p1, t);
  const int s_hi = tf::exact::parameter_plane_sign<int64>(
      pl_hi[0], pl_hi[1], pl_hi[2], p0, p1, t);
  REQUIRE(s_lo != 0);
  REQUIRE(s_hi != 0);
  REQUIRE(s_lo == -s_hi);
}

TEST_CASE("edge_parameter: the geological junction shape - segment in the "
          "wall, crossing exactly on the wall",
          "[exact][edge_parameter]") {
  // The witness configuration: a boundary edge lying IN the wall plane
  // x = 0, a horizon plane crossing it. The crossing point must test
  // exactly ON the wall — an incidence a snapped coordinate cannot carry.
  const pt p0{0, 0, 0}, p1{0, 10, 4};
  const std::array<pt, 3> wall{pt{0, 1, 5}, pt{0, -4, 2}, pt{0, 9, -3}};
  const std::array<pt, 3> horizon{pt{-5, 3, 0}, pt{7, 3, 1}, pt{2, 3, 9}};
  const auto t = tf::exact::make_edge_plane_parameter<int64>(
      horizon[0], horizon[1], horizon[2], p0, p1);
  REQUIRE(tf::exact::parameter_plane_sign<int64>(wall[0], wall[1], wall[2],
                                                 p0, p1, t) == 0);
}

TEST_CASE("edge_parameter: EE and EF constructions of one point agree",
          "[exact][edge_parameter]") {
  // Two segments crossing in the z = 0 plane; the second segment also
  // spans a vertical plane. The crossing via segment x segment and via
  // segment x plane is the same point: the fractions must compare 0.
  const pt p0{0, 0, 0}, p1{10, 0, 0};
  const pt q0{5, -3, 0}, q1{5, 7, 0};
  const std::array<pt, 3> qplane{pt{5, -3, 0}, pt{5, 7, 0}, pt{5, 0, 9}};
  const auto axes = tf::exact::projection_axes(pt{0, 0, 0}, pt{10, 0, 0},
                                               pt{0, 10, 0});
  const auto t_ee =
      tf::exact::make_edge_edge_parameter<int64>(p0, p1, q0, q1, axes);
  const auto t_ef = tf::exact::make_edge_plane_parameter<int64>(
      qplane[0], qplane[1], qplane[2], p0, p1);
  REQUIRE(tf::exact::compare_parameter(t_ee, t_ef) == 0);
}

TEST_CASE("edge_parameter: shear invariance of order and incidence",
          "[exact][edge_parameter]") {
  std::mt19937_64 rng(0x6b31d20fu);
  const auto coord = [&]() -> int64 { return int64(rng() % 2001) - 1000; };
  const auto shear = [&](const pt &p, int64 kxy, int64 kyz,
                         int64 kzx) -> pt {
    // unimodular: volume signs and segment ratios are preserved
    const int64 x = p[0] + kxy * p[1];
    const int64 y = p[1] + kyz * p[2];
    const int64 z = p[2] + kzx * x;
    return pt{x, y, z};
  };
  for (int it = 0; it < 500; ++it) {
    const pt p0{coord(), coord(), coord()};
    pt p1{coord(), coord(), coord()};
    if (p0[0] == p1[0] && p0[1] == p1[1] && p0[2] == p1[2])
      p1[0] = p1[0] + 1;
    std::array<pt, 3> pl_a{pt{coord(), coord(), coord()},
                           pt{coord(), coord(), coord()},
                           pt{coord(), coord(), coord()}};
    std::array<pt, 3> pl_b{pt{coord(), coord(), coord()},
                           pt{coord(), coord(), coord()},
                           pt{coord(), coord(), coord()}};
    const auto t_a = tf::exact::make_edge_plane_parameter<int64>(
        pl_a[0], pl_a[1], pl_a[2], p0, p1);
    const auto t_b = tf::exact::make_edge_plane_parameter<int64>(
        pl_b[0], pl_b[1], pl_b[2], p0, p1);
    if (t_a.den == decltype(t_a.den)(0) || t_b.den == decltype(t_b.den)(0))
      continue; // parallel draw: outside the factory contract
    const int order = tf::exact::compare_parameter(t_a, t_b);
    const int side = tf::exact::parameter_plane_sign<int64>(
        pl_b[0], pl_b[1], pl_b[2], p0, p1, t_a);
    const int64 kxy = int64(rng() % 7) - 3, kyz = int64(rng() % 7) - 3,
                kzx = int64(rng() % 7) - 3;
    const auto s = [&](const pt &p) { return shear(p, kxy, kyz, kzx); };
    const auto t_a2 = tf::exact::make_edge_plane_parameter<int64>(
        s(pl_a[0]), s(pl_a[1]), s(pl_a[2]), s(p0), s(p1));
    const auto t_b2 = tf::exact::make_edge_plane_parameter<int64>(
        s(pl_b[0]), s(pl_b[1]), s(pl_b[2]), s(p0), s(p1));
    REQUIRE(tf::exact::compare_parameter(t_a2, t_b2) == order);
    REQUIRE(tf::exact::parameter_plane_sign<int64>(s(pl_b[0]), s(pl_b[1]),
                                                   s(pl_b[2]), s(p0), s(p1),
                                                   t_a2) == side);
    REQUIRE(tf::exact::compare_parameter(t_a, t_a) == 0);
  }
}

TEST_CASE("edge_parameter: dyadic materialization lands on the fraction",
          "[exact][edge_parameter]") {
  const pt p0{0, 0, 0}, p1{16, 0, 0};
  const auto pl = x_plane(4, 5);
  const auto t = tf::exact::make_edge_plane_parameter<int64>(pl[0], pl[1],
                                                             pl[2], p0, p1);
  const auto bits = 16;
  const auto dy = tf::exact::dyadic_ratio<int64>(t.num, t.den, bits);
  REQUIRE(dy == (typename tf::exact::meta<int64>::param_type(1) << bits) / 4);
}

TEST_CASE("edge_parameter: la's crossing construction IS the emission's "
          "fraction — the junction twin is unrepresentable",
          "[exact][edge_parameter]") {
  // The geological junction, reduced: edge e4 lies IN the wall plane;
  // the (5,10) seam is the line wall ∩ horizon. The seam crossing e4,
  // computed on generators, must select the non-degenerate defining
  // plane (e4 ⊂ wall makes the wall's denominator exactly zero) and
  // then produce THE SAME fraction the (4,5) emission stored — bit for
  // bit, because it is the same construction on the same inputs. No
  // comparison tolerance exists anywhere in this chain.
  const pt p0{0, 0, 0}, p1{0, 10, 4};                       // e4, in x=0
  const std::array<pt, 3> wall{pt{0, 1, 5}, pt{0, -4, 2}, pt{0, 9, -3}};
  const std::array<pt, 3> horizon{pt{-5, 3, 0}, pt{7, 3, 1}, pt{2, 3, 9}};

  // Emission's fraction: e4 x horizon (the (4,5) EF record).
  const auto t_emitted = tf::exact::make_edge_plane_parameter<int64>(
      horizon[0], horizon[1], horizon[2], p0, p1);

  // The local arrangement's crossing of e4 with the seam line (wall ∩ horizon):
  // the wall's denominator vanishes exactly — the degeneracy detector —
  // so the construction falls to the horizon plane.
  const auto vol_w0 = tf::exact::orient3d_value(wall[0], wall[1], wall[2], p0);
  const auto vol_w1 = tf::exact::orient3d_value(wall[0], wall[1], wall[2], p1);
  REQUIRE(vol_w0 - vol_w1 == decltype(vol_w0)(0)); // e4 ⊂ wall, exactly
  const auto t_crossing = tf::exact::make_edge_plane_parameter<int64>(
      horizon[0], horizon[1], horizon[2], p0, p1);

  REQUIRE(t_crossing.num == t_emitted.num); // same fraction, bit-level
  REQUIRE(t_crossing.den == t_emitted.den);
  REQUIRE(tf::exact::compare_parameter(t_crossing, t_emitted) == 0);
  // and the point at that parameter is exactly ON the wall — the
  // incidence a snapped coordinate cannot carry:
  REQUIRE(tf::exact::parameter_plane_sign<int64>(wall[0], wall[1], wall[2],
                                                 p0, p1, t_crossing) == 0);
}

TEST_CASE("edge_parameter: parameter_line_sign2 matches the materialized "
          "orientation and lands exact zeros",
          "[exact]") {
  using pt2 = tf::exact::pt2<int64>;
  std::mt19937_64 rng(2026);
  std::uniform_int_distribution<int64> coord(-1000000, 1000000);
  for (int it = 0; it < 2000; ++it) {
    const pt2 a{coord(rng), coord(rng)}, b{coord(rng), coord(rng)};
    const pt2 p0{coord(rng), coord(rng)}, p1{coord(rng), coord(rng)};
    std::uniform_int_distribution<int64> den_d(1, 1 << 20);
    const auto den = den_d(rng);
    std::uniform_int_distribution<int64> num_d(0, den);
    const auto num = num_d(rng);
    const tf::exact::edge_parameter<int64> t{num, den};
    // oracle: den * O(p0) + num * (O(p1) - O(p0)) in wide arithmetic
    const auto o_0 = tf::exact::orient2d(a, b, p0);
    const auto o_1 = tf::exact::orient2d(a, b, p1);
    const auto lhs = decltype(o_0)(den) * o_0 + decltype(o_0)(num) * (o_1 - o_0);
    const int expected = lhs > decltype(lhs)(0)   ? 1
                         : lhs < decltype(lhs)(0) ? -1
                                                  : 0;
    REQUIRE(tf::exact::parameter_line_sign2<int64>(a, b, p0, p1, t) ==
            expected);
  }
  // exact incidence: the crossing of (p0,p1) with line (a,b), stated as
  // the segment-segment fraction, sits ON the line — sign exactly zero.
  const pt a3{-7, -3, 0}, b3{11, 5, 0};
  const pt q0{-2, 9, 0}, q1{4, -13, 0};
  const auto t =
      tf::exact::make_edge_edge_parameter<int64>(q0, q1, a3, b3, {0, 1});
  REQUIRE(t.den > 0);
  const pt2 a{a3[0], a3[1]}, b{b3[0], b3[1]};
  const pt2 p0{q0[0], q0[1]}, p1{q1[0], q1[1]};
  REQUIRE(tf::exact::parameter_line_sign2<int64>(a, b, p0, p1, t) == 0);
}

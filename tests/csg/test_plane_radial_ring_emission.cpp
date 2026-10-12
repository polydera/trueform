/**
 * @file test_plane_radial_ring_emission.cpp
 * @brief The ring certificate reads the emitted half-planes against the
 * carriers
 *
 * A ring's order is computed on its carriers and published only where the
 * emitted surface stands in it. The emitted half-planes may wind around the
 * piece in the carriers' order and still disagree with them: a piece the
 * rounding left off one of its carriers turns the emitted half-planes with
 * it, and a page's apex can stand across another page's carrier while its
 * half-plane keeps its place in the ring. That ring certifies no sector.
 *
 * The ring is stated on the lattice by hand: carriers P (z = 0), Q
 * (-y + 1000 z = 0) and R (y = -3000), the piece on Q and R three units off
 * P, and Q's apex on Q, below P, though Q's page rises above it.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/csg/graph/read_plane_radial_ring_emission.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/offset_dot.hpp>
#include <trueform/exact/projected_dot.hpp>
#include <trueform/exact/triple_product.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace {

template <typename Int> struct astray_ring {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using carrier_t = tf::csg::graph::plane_radial_ring_carrier<int, Int>;

  std::vector<std::array<T1, 3>> halves;
  std::array<T1, 3> line{T1(1000), T1(0), T1(0)};
  std::vector<carrier_t> carriers;

  // `q_apex_z` is Q's apex's height over P, its position on Q following
  astray_ring(int q_apex_z) {
    const std::array<std::int64_t, 3> start{0, -3000, -3};
    const std::array<std::array<std::int64_t, 3>, 3> apexes{
        {{{500, 5000, 0}},
         {{500, 1000 * q_apex_z, q_apex_z}},
         {{500, -3000, 1000}}}};
    for (const auto &apex : apexes)
      halves.push_back({T1(apex[0] - start[0]), T1(apex[1] - start[1]),
                        T1(apex[2] - start[2])});
    const auto carrier = [&](std::array<std::int64_t, 3> normal,
                             std::int64_t offset, int plane, signed char side) {
      const std::array<T2, 3> n{T2(normal[0]), T2(normal[1]), T2(normal[2])};
      const T2 residual = T2(normal[0] * start[0] + normal[1] * start[1] +
                             normal[2] * start[2] - offset);
      return carrier_t{n,
                       residual,
                       tf::csg::graph::plane_radial_ring_image(n),
                       double(residual),
                       plane,
                       side};
    };
    // in ring order: P's page along +y, Q's just above it, R's along +z
    carriers.push_back(carrier({0, 0, 1}, 0, 0, 1));
    carriers.push_back(carrier({0, -1, 1000}, 0, 1, 1));
    carriers.push_back(carrier({0, 1, 0}, -3000, 2, -1));
  }
};

template <typename Int> auto read_astray_ring(int q_apex_z) {
  const astray_ring<Int> ring(q_apex_z);
  std::vector<std::array<double, 3>> halves_image;
  for (const auto &half : ring.halves)
    halves_image.push_back(tf::csg::graph::plane_radial_ring_image(half));
  return tf::csg::graph::read_plane_radial_ring_emission<Int>(
      ring.halves, halves_image, ring.line, ring.carriers);
}

} // namespace

TEMPLATE_TEST_CASE("plane radial ring: an apex across another carrier",
                   "[arrangement][planes][radial][certificate]",
                   tf::exact::int32, tf::exact::int64) {
  // Q's apex one unit below P: the half-planes wind in the carriers' order,
  // but the apex stands across P from where Q's page rises
  const auto astray = read_astray_ring<TestType>(-1);
  CHECK(!astray.contradicted);
  CHECK(!astray.tied);
  CHECK(!astray.flat);
  CHECK(astray.astray);
  // one unit above P it stands where Q's page puts it
  const auto placed = read_astray_ring<TestType>(1);
  CHECK(!placed.contradicted);
  CHECK(!placed.tied);
  CHECK(!placed.flat);
  CHECK(!placed.astray);
}

TEST_CASE("plane radial ring: the readings defer to the exact rung",
          "[arrangement][planes][radial][certificate]") {
  // operands at the int64 lattice's scale whose readings cancel to a unit,
  // so no double filter can decide them and the exact rung answers
  using T1 = tf::exact::meta<tf::exact::int64>::T1;
  using T2 = tf::exact::meta<tf::exact::int64>::T2;
  using T3 = tf::exact::meta<tf::exact::int64>::T3;
  const T1 big = T1(std::int64_t(1) << 60);
  const std::array<T1, 3> a{big, T1(1), T1(0)};
  const std::array<T1, 3> b{big + T1(1), T1(1), T1(0)};
  const std::array<T1, 3> up{T1(0), T1(0), T1(1)};
  // (a x b) . up = big - (big + 1)
  CHECK(tf::exact::triple_product::filter_sign(a, b, up) == 0);
  CHECK(tf::exact::triple_product_sign<T2>(a, b, up) == -1);
  // |a|^2 |b|^2 - (a . b)^2 = |a x b|^2 = 1: a and b are not parallel
  CHECK(tf::exact::projected_dot::filter_sign(a, a, b) == 0);
  CHECK(tf::exact::projected_dot_sign<T3>(a, a, b) == 1);
  // (big, big + 1, 0) . (1, -1, 0) + 1 = 0, and + 2 = 1
  const std::array<T2, 3> n{T2(big), T2(big + T1(1)), T2(0)};
  const std::array<T1, 3> h{T1(1), T1(-1), T1(0)};
  CHECK(tf::exact::offset_dot::filter_sign(n, h, T2(1)) == 0);
  CHECK(tf::exact::offset_dot_sign<T2>(n, h, T2(1)) == 0);
  CHECK(tf::exact::offset_dot_sign<T2>(n, h, T2(2)) == 1);
}

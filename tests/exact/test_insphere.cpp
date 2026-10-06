/**
 * @file test_insphere.cpp
 * @brief tf::exact::insphere — the sphere a tet names, and who is inside it.
 *
 * The sign convention, the total antisymmetry of the five rows, the rungs
 * the determinant needs at the lattice extremes and at scale, and the
 * lift-only SoS cascade against an independently expanded perturbed
 * determinant.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/insphere.hpp>
#include <trueform/exact/insphere/filter_sign.hpp>
#include <trueform/exact/int256.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int512.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/orient3d.hpp>
#include <trueform/exact/vertex.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>

namespace {

using insphere_i32 = tf::exact::int32;
using insphere_i64 = tf::exact::int64;
using insphere_pt32 = tf::exact::pt3<insphere_i32>;
using insphere_pt64 = tf::exact::pt3<insphere_i64>;
using insphere_value32 = tf::exact::meta<insphere_i32>::T3;
using insphere_value64 = tf::exact::meta<insphere_i64>::T3;
using insphere_site32 = tf::exact::vertex<int, insphere_i32>;

constexpr insphere_i32 insphere_max32 = 2147483647;
constexpr insphere_i64 insphere_max64 = 9223372036854775807LL;

template <typename T> auto insphere_from_decimal(const char *text) -> T {
  const bool negative = (*text == '-');
  if (negative)
    ++text;
  T value(0);
  for (; *text != '\0'; ++text)
    value = value * T(10) + T(*text - '0');
  return negative ? -value : value;
}

auto insphere_site(int id, insphere_i32 x, insphere_i32 y, insphere_i32 z)
    -> insphere_site32 {
  return insphere_site32{id, insphere_pt32{x, y, z}};
}

auto insphere_scaled_pt(const insphere_pt32 &p, insphere_i64 scale)
    -> insphere_pt64 {
  return insphere_pt64{insphere_i64(p[0]) * scale, insphere_i64(p[1]) * scale,
                       insphere_i64(p[2]) * scale};
}

// The oracle: the five-site determinant expanded by Leibniz over its 120
// permutations, with the lift column carrying `|p|^2 - eps^(2^rank)` as an
// integer polynomial in eps. It shares no step with the predicate's
// cofactor expansion; the verdict is the sign of the lowest-degree nonzero
// coefficient of MINUS that determinant, which is what eps -> 0+ states.
constexpr std::size_t insphere_poly_terms = 17;

struct insphere_poly {
  std::array<long long, insphere_poly_terms> c{};
  std::size_t width = 1;
};

auto insphere_poly_mul(const insphere_poly &p, const insphere_poly &q)
    -> insphere_poly {
  insphere_poly r{};
  for (std::size_t i = 0; i < p.width; ++i)
    if (p.c[i] != 0)
      for (std::size_t j = 0; j < q.width && i + j < insphere_poly_terms; ++j)
        r.c[i + j] += p.c[i] * q.c[j];
  const std::size_t width = p.width + q.width - 1;
  r.width = width < insphere_poly_terms ? width : insphere_poly_terms;
  return r;
}

auto insphere_poly_constant(long long v) -> insphere_poly {
  insphere_poly p{};
  p.c[0] = v;
  return p;
}

auto insphere_permutation_parity(const std::array<int, 5> &perm) -> int {
  int parity = 1;
  for (int i = 0; i < 5; ++i)
    for (int j = i + 1; j < 5; ++j)
      if (perm[std::size_t(i)] > perm[std::size_t(j)])
        parity = -parity;
  return parity;
}

template <typename Pt>
auto insphere_perturbed_sign(const std::array<Pt, 5> &pts,
                             const std::array<int, 5> &ranks) -> int {
  std::array<int, 5> order = {0, 1, 2, 3, 4};
  std::sort(order.begin(), order.end(), [&ranks](int a, int b) {
    return ranks[std::size_t(a)] < ranks[std::size_t(b)];
  });
  std::array<int, 5> compressed{};
  for (int k = 0; k < 5; ++k)
    compressed[std::size_t(order[std::size_t(k)])] = k;

  std::array<std::array<insphere_poly, 5>, 5> m{};
  for (std::size_t row = 0; row < 5; ++row) {
    const auto &p = pts[row];
    for (std::size_t axis = 0; axis < 3; ++axis)
      m[row][axis] = insphere_poly_constant(p[axis]);
    insphere_poly lift = insphere_poly_constant(
        (long long)(p[0]) * p[0] + (long long)(p[1]) * p[1] +
        (long long)(p[2]) * p[2]);
    const std::size_t degree = std::size_t(1) << compressed[row];
    lift.c[degree] -= 1;
    lift.width = degree + 1;
    m[row][3] = lift;
    m[row][4] = insphere_poly_constant(1);
  }

  insphere_poly total{};
  std::array<int, 5> perm = {0, 1, 2, 3, 4};
  do {
    insphere_poly term =
        insphere_poly_constant(insphere_permutation_parity(perm));
    for (std::size_t row = 0; row < 5; ++row)
      term = insphere_poly_mul(term, m[row][std::size_t(perm[row])]);
    for (std::size_t k = 0; k < insphere_poly_terms; ++k)
      total.c[k] += term.c[k];
  } while (std::next_permutation(perm.begin(), perm.end()));

  for (std::size_t k = 0; k < insphere_poly_terms; ++k)
    if (total.c[k] != 0)
      return total.c[k] < 0 ? 1 : -1;
  return 0;
}

auto insphere_cube_sites() -> std::array<insphere_pt32, 8> {
  return {insphere_pt32{0, 0, 0}, insphere_pt32{2, 0, 0},
          insphere_pt32{0, 2, 0}, insphere_pt32{2, 2, 0},
          insphere_pt32{0, 0, 2}, insphere_pt32{2, 0, 2},
          insphere_pt32{0, 2, 2}, insphere_pt32{2, 2, 2}};
}

} // namespace

TEST_CASE("insphere: the unit tet witness fixes the sign convention",
          "[exact][insphere]") {
  const insphere_pt32 a{0, 0, 0}, b{4, 0, 0}, c{0, 4, 0}, d{0, 0, 4};

  REQUIRE(tf::exact::orient3d_value<insphere_i32>(a, b, c, d) == 64);

  // The circumsphere is centred at (2, 2, 2) with radius squared 12.
  const insphere_pt32 inside{1, 1, 1};
  const insphere_pt32 on{4, 4, 4};
  const insphere_pt32 outside{5, 5, 5};

  CHECK(tf::exact::insphere_value<insphere_i32>(a, b, c, d, inside) ==
        insphere_value32(576));
  CHECK(tf::exact::insphere_sign<insphere_i32>(a, b, c, d, inside) == 1);
  CHECK(tf::exact::insphere_sign<insphere_i32>(a, b, c, d, on) == 0);
  CHECK(tf::exact::insphere_sign<insphere_i32>(a, b, c, d, outside) == -1);

  // The same tet wound the other way reverses both facts together.
  REQUIRE(tf::exact::orient3d_value<insphere_i32>(a, c, b, d) == -64);
  CHECK(tf::exact::insphere_value<insphere_i32>(a, c, b, d, inside) ==
        insphere_value32(-576));
  CHECK(tf::exact::insphere_sign<insphere_i32>(a, c, b, d, outside) == 1);
}

TEST_CASE("insphere: the value is antisymmetric in all five sites",
          "[exact][insphere]") {
  const std::array<insphere_pt32, 5> sites = {
      insphere_pt32{0, 0, 0}, insphere_pt32{4, 0, 0}, insphere_pt32{0, 4, 0},
      insphere_pt32{0, 0, 4}, insphere_pt32{1, 1, 1}};
  const auto base = tf::exact::insphere_value<insphere_i32>(
      sites[0], sites[1], sites[2], sites[3], sites[4]);

  std::array<int, 5> perm = {0, 1, 2, 3, 4};
  int checked = 0;
  do {
    const auto value = tf::exact::insphere_value<insphere_i32>(
        sites[std::size_t(perm[0])], sites[std::size_t(perm[1])],
        sites[std::size_t(perm[2])], sites[std::size_t(perm[3])],
        sites[std::size_t(perm[4])]);
    const auto expected = insphere_permutation_parity(perm) > 0 ? base : -base;
    REQUIRE(value == expected);
    ++checked;
  } while (std::next_permutation(perm.begin(), perm.end()));
  CHECK(checked == 120);
}

TEST_CASE("insphere: the conflict verdict survives relabelling the tet",
          "[exact][insphere]") {
  const std::array<insphere_site32, 4> tet = {
      insphere_site(0, 0, 0, 0), insphere_site(1, 4, 0, 0),
      insphere_site(2, 0, 4, 0), insphere_site(3, 0, 0, 4)};
  const std::array<insphere_site32, 3> queries = {
      insphere_site(4, 1, 1, 1), // strictly inside the circumsphere
      insphere_site(4, 5, 5, 5), // strictly outside
      insphere_site(4, 4, 4, 4)  // cospherical, decided by the cascade
  };
  const std::array<bool, 3> expected = {true, false, true};

  for (std::size_t q = 0; q < queries.size(); ++q) {
    std::array<int, 4> perm = {0, 1, 2, 3};
    do {
      std::array<insphere_site32, 5> sites;
      for (std::size_t i = 0; i < 4; ++i)
        sites[i] = tet[std::size_t(perm[i])];
      sites[4] = queries[q];
      const int orientation = tf::exact::orient3d_sign<insphere_i32>(
          sites[0].pt, sites[1].pt, sites[2].pt, sites[3].pt);
      REQUIRE(orientation != 0);
      CHECK(tf::exact::insphere_conflict(sites, orientation) == expected[q]);
    } while (std::next_permutation(perm.begin(), perm.end()));
  }
}

TEST_CASE("insphere: the int32 lattice extremes stay inside the int256 rung",
          "[exact][insphere]") {
  // python3 - <<'EOF'
  // M = (1 << 31) - 1
  // def det(m):
  //     n = len(m)
  //     if n == 1: return m[0][0]
  //     return sum((-1)**j * m[0][j] * det([r[:j]+r[j+1:] for r in m[1:]])
  //                for j in range(n))
  // def insphere(a, b, c, d, e):
  //     rows = [[x[i]-e[i] for i in range(3)] for x in (a, b, c, d)]
  //     return -det([r + [r[0]**2 + r[1]**2 + r[2]**2] for r in rows])
  // print(insphere((-M,-M,M), (-M,M,-M), (M,-M,-M), (M,M,M), (0,0,0)))
  // print(insphere((-M,-M,-M), (M,-M,-M), (-M,M,-M), (-M,-M,M), (0,0,0)))
  // EOF
  const insphere_i32 m = insphere_max32;

  const insphere_pt32 a{-m, -m, m}, b{-m, m, -m}, c{m, -m, -m}, d{m, m, m};
  const insphere_pt32 centre{0, 0, 0};
  const auto corner_value =
      tf::exact::insphere_value<insphere_i32>(a, b, c, d, centre);
  CHECK(corner_value ==
        insphere_from_decimal<insphere_value32>(
            "-2192252450892118878245140045766056385290262544336"));
  // The four corners are equidistant from the origin, so the centre of
  // their circumsphere is the query: it conflicts either way the tet winds.
  const int corner_orientation =
      tf::exact::orient3d_sign<insphere_i32>(a, b, c, d);
  CHECK(corner_orientation == -1);
  CHECK(tf::exact::insphere_sign<insphere_i32>(a, b, c, d, centre) == -1);

  const insphere_pt32 p0{-m, -m, -m}, p1{m, -m, -m}, p2{-m, m, -m},
      p3{-m, -m, m};
  CHECK(tf::exact::insphere_value<insphere_i32>(p0, p1, p2, p3, centre) ==
        insphere_from_decimal<insphere_value32>(
            "1096126225446059439122570022883028192645131272168"));
  CHECK(tf::exact::orient3d_sign<insphere_i32>(p0, p1, p2, p3) == 1);
  CHECK(tf::exact::insphere_sign<insphere_i32>(p0, p1, p2, p3, centre) == 1);
}

TEST_CASE("insphere: the int64 lattice needs the rung past meta::T2",
          "[exact][insphere]") {
  // The same script at M = (1 << 63) - 1; the magnitude is 321 bits, so
  // meta<int64>::T2 (int256) cannot carry it and int512 must.
  static_assert(tf::exact::insphere_bits(
                    tf::exact::meta<insphere_i64>::coordinate_bits) >
                    tf::exact::meta<insphere_i64>::t2_bits,
                "the int64 insphere value must outgrow its own T2 rung");

  const insphere_i64 m = insphere_max64;
  const insphere_pt64 a{-m, -m, m}, b{-m, m, -m}, c{m, -m, -m}, d{m, m, m};
  const insphere_pt64 centre{0, 0, 0};
  CHECK(tf::exact::insphere_value<insphere_i64>(a, b, c, d, centre) ==
        insphere_from_decimal<insphere_value64>(
            "-320398055388136512185565122069458524092711811252786138291082612"
            "3110668497431847047980621217398736"));
  CHECK(tf::exact::orient3d_sign<insphere_i64>(a, b, c, d) == -1);
  CHECK(tf::exact::insphere_sign<insphere_i64>(a, b, c, d, centre) == -1);

  // The cascade at this lattice reads its orientation minors on int256.
  const std::array<insphere_pt64, 5> cospherical = {
      insphere_pt64{-2, -2, -2}, insphere_pt64{2, -2, -2},
      insphere_pt64{-2, 2, -2}, insphere_pt64{-2, -2, 2},
      insphere_pt64{2, 2, 2}};
  REQUIRE(tf::exact::insphere_sign<insphere_i64>(
              cospherical[0], cospherical[1], cospherical[2], cospherical[3],
              cospherical[4]) == 0);
  std::array<int, 5> ranks = {0, 1, 2, 3, 4};
  do {
    std::array<tf::exact::vertex<int, insphere_i64>, 5> sites;
    for (std::size_t k = 0; k < 5; ++k)
      sites[k] = {ranks[k], cospherical[k]};
    REQUIRE(tf::exact::insphere_sos(sites) ==
            insphere_perturbed_sign(cospherical, ranks));
  } while (std::next_permutation(ranks.begin(), ranks.end()));

  const std::array<tf::exact::vertex<int, insphere_i64>, 5> sites = {
      tf::exact::vertex<int, insphere_i64>{0, cospherical[0]},
      tf::exact::vertex<int, insphere_i64>{1, cospherical[1]},
      tf::exact::vertex<int, insphere_i64>{2, cospherical[2]},
      tf::exact::vertex<int, insphere_i64>{3, cospherical[3]},
      tf::exact::vertex<int, insphere_i64>{4, cospherical[4]}};
  const int orientation = tf::exact::orient3d_sign<insphere_i64>(
      cospherical[0], cospherical[1], cospherical[2], cospherical[3]);
  REQUIRE(orientation == 1);
  CHECK(tf::exact::insphere_conflict(sites, orientation) ==
        (tf::exact::insphere_sos(sites) > 0));
}

TEST_CASE("insphere: every rung admits the scaled lattice the door caps",
          "[exact][insphere]") {
  static_assert(tf::exact::insphere_scale_bits<insphere_i32>() >= 6,
                "the int32 lattice must serve a depth-six protection scale");
  static_assert(tf::exact::insphere_scale_bits<insphere_i64>() >= 6,
                "the int64 lattice must serve a depth-six protection scale");

  // The published headroom is the smallest of the rungs' own: a 3x3 minor
  // of differences on T2 gives out first at both lattices.
  CHECK(tf::exact::insphere_scale_bits<insphere_i32>() == 9);
  CHECK(tf::exact::insphere_scale_bits<insphere_i64>() == 20);

  CHECK(tf::exact::insphere_minor_bits(37) == 117);
  CHECK(tf::exact::insphere_bits(37) + 1 == 198);
  CHECK(tf::exact::insphere_minor_bits(69) == 213);
  CHECK(tf::exact::insphere_bits(69) + 1 == 358);
}

TEST_CASE("insphere: the scaled entry reads the same witnesses at scale 64",
          "[exact][insphere]") {
  const insphere_i64 scale = 64;

  const insphere_pt32 a{0, 0, 0}, b{4, 0, 0}, c{0, 4, 0}, d{0, 0, 4},
      inside{1, 1, 1}, on{4, 4, 4}, outside{5, 5, 5};
  const auto sa = insphere_scaled_pt(a, scale);
  const auto sb = insphere_scaled_pt(b, scale);
  const auto sc = insphere_scaled_pt(c, scale);
  const auto sd = insphere_scaled_pt(d, scale);

  // The value scales with the fifth power of the common multiple.
  CHECK(tf::exact::insphere_value_scaled<insphere_i32>(
            sa, sb, sc, sd, insphere_scaled_pt(inside, scale)) ==
        insphere_value32(576) * insphere_value32(1073741824));
  CHECK(tf::exact::insphere_sign_scaled<insphere_i32>(
            sa, sb, sc, sd, insphere_scaled_pt(on, scale)) == 0);
  CHECK(tf::exact::insphere_sign_scaled<insphere_i32>(
            sa, sb, sc, sd, insphere_scaled_pt(outside, scale)) == -1);

  // The lattice extremes at scale six: 37 value bits per coordinate, stored
  // in meta<int32>::T1 while the widths still dispatch from int32.
  const insphere_i32 m = insphere_max32;
  const auto xa = insphere_scaled_pt(insphere_pt32{-m, -m, m}, scale);
  const auto xb = insphere_scaled_pt(insphere_pt32{-m, m, -m}, scale);
  const auto xc = insphere_scaled_pt(insphere_pt32{m, -m, -m}, scale);
  const auto xd = insphere_scaled_pt(insphere_pt32{m, m, m}, scale);
  const insphere_pt64 centre{0, 0, 0};
  CHECK(tf::exact::insphere_value_scaled<insphere_i32>(xa, xb, xc, xd,
                                                       centre) ==
        insphere_from_decimal<insphere_value32>(
            "-2353913145289374151551770591876288860428413273794217508864"));
  CHECK(tf::exact::insphere_sign_scaled<insphere_i32>(xa, xb, xc, xd,
                                                      centre) == -1);
}

TEST_CASE("insphere: the lift cascade matches a perturbed determinant",
          "[exact][insphere]") {
  const auto cube = insphere_cube_sites();
  // Every rank assignment of every cospherical five-subset of a cube, each
  // decided twice: by the cascade, and by the independently expanded
  // perturbed determinant.
  std::array<int, 5> ranks = {0, 1, 2, 3, 4};
  int checked = 0;
  int cascade_used = 0;
  for (int i0 = 0; i0 < 8; ++i0)
    for (int i1 = i0 + 1; i1 < 8; ++i1)
      for (int i2 = i1 + 1; i2 < 8; ++i2)
        for (int i3 = i2 + 1; i3 < 8; ++i3)
          for (int i4 = i3 + 1; i4 < 8; ++i4) {
            const std::array<int, 5> pick = {i0, i1, i2, i3, i4};
            std::array<insphere_pt32, 5> pts;
            for (std::size_t k = 0; k < 5; ++k)
              pts[k] = cube[std::size_t(pick[k])];
            if (tf::exact::insphere_sign<insphere_i32>(pts[0], pts[1], pts[2],
                                                       pts[3], pts[4]) == 0)
              ++cascade_used;
            ranks = {0, 1, 2, 3, 4};
            do {
              std::array<insphere_site32, 5> sites;
              for (std::size_t k = 0; k < 5; ++k)
                sites[k] = insphere_site32{ranks[k], pts[k]};
              REQUIRE(tf::exact::insphere_sos(sites) ==
                      insphere_perturbed_sign(pts, ranks));
              ++checked;
            } while (std::next_permutation(ranks.begin(), ranks.end()));
          }
  CHECK(checked == 56 * 120);
  CHECK(cascade_used == 56);
}

TEST_CASE("insphere: a cospherical query is decided by its own term",
          "[exact][insphere]") {
  // Every five-subset of a cube's corners is cospherical, so each one
  // reaches the cascade. The query is row 4 and its coefficient is the
  // tet's own orientation: with a nondegenerate tet the walk always
  // decides, and when the query also carries the lowest rank its term is
  // read first, which is what "lowering a site's own lift makes it more
  // inside" states.
  const auto cube = insphere_cube_sites();
  int decided = 0;
  for (int i0 = 0; i0 < 8; ++i0)
    for (int i1 = 0; i1 < 8; ++i1)
      for (int i2 = 0; i2 < 8; ++i2)
        for (int i3 = 0; i3 < 8; ++i3)
          for (int i4 = 0; i4 < 8; ++i4) {
            const std::array<int, 5> pick = {i0, i1, i2, i3, i4};
            std::array<int, 5> seen = pick;
            std::sort(seen.begin(), seen.end());
            if (std::adjacent_find(seen.begin(), seen.end()) != seen.end())
              continue;
            std::array<insphere_site32, 5> sites;
            for (std::size_t k = 0; k < 5; ++k)
              sites[k] =
                  insphere_site32{int(k) + 1, cube[std::size_t(pick[k])]};
            const int orientation = tf::exact::orient3d_sign<insphere_i32>(
                sites[0].pt, sites[1].pt, sites[2].pt, sites[3].pt);
            if (orientation == 0)
              continue;
            REQUIRE(tf::exact::insphere_sign<insphere_i32>(
                        sites[0].pt, sites[1].pt, sites[2].pt, sites[3].pt,
                        sites[4].pt) == 0);
            REQUIRE(tf::exact::insphere_sos(sites) != 0);
            sites[4].id = 0;
            CHECK(tf::exact::insphere_conflict(sites, orientation));
            ++decided;
          }
  CHECK(decided == 5568);
}

TEST_CASE("insphere: the six-site witness gives both neighbours one verdict",
          "[exact][insphere]") {
  // a, b, c, u, v, p with p cocircular with face abc: the two cells sharing
  // abc must not disagree about p, or abc becomes a cavity boundary and
  // starring it stores a flat tet.
  const auto a = insphere_site(0, 0, 0, 0);
  const auto b = insphere_site(1, 2, 0, 0);
  const auto c = insphere_site(2, 0, 2, 0);
  const auto u = insphere_site(3, 1, 1, 3);
  const auto v = insphere_site(4, 1, 1, -3);
  const auto p = insphere_site(5, 2, 2, 0);

  REQUIRE(tf::exact::orient3d_value<insphere_i32>(a.pt, b.pt, c.pt, u.pt) ==
          12);
  REQUIRE(tf::exact::orient3d_value<insphere_i32>(a.pt, b.pt, c.pt, v.pt) ==
          -12);
  REQUIRE(tf::exact::insphere_sign<insphere_i32>(a.pt, b.pt, c.pt, u.pt,
                                                 p.pt) == 0);
  REQUIRE(tf::exact::insphere_sign<insphere_i32>(a.pt, b.pt, c.pt, v.pt,
                                                 p.pt) == 0);

  // The five cascade coefficients in row order, each `(-1)^row` times the
  // orientation of the other four rows left in place. The apex coefficient
  // is zero on both sides and a's decides, so both normalized polynomials
  // are the same `e_a - e_b - e_c + e_p`.
  CHECK(+tf::exact::orient3d_value<insphere_i32>(b.pt, c.pt, u.pt, p.pt) == 12);
  CHECK(-tf::exact::orient3d_value<insphere_i32>(a.pt, c.pt, u.pt, p.pt) ==
        -12);
  CHECK(+tf::exact::orient3d_value<insphere_i32>(a.pt, b.pt, u.pt, p.pt) ==
        -12);
  CHECK(-tf::exact::orient3d_value<insphere_i32>(a.pt, b.pt, c.pt, p.pt) == 0);
  CHECK(+tf::exact::orient3d_value<insphere_i32>(a.pt, b.pt, c.pt, u.pt) == 12);

  CHECK(+tf::exact::orient3d_value<insphere_i32>(b.pt, c.pt, v.pt, p.pt) ==
        -12);
  CHECK(-tf::exact::orient3d_value<insphere_i32>(a.pt, c.pt, v.pt, p.pt) == 12);
  CHECK(+tf::exact::orient3d_value<insphere_i32>(a.pt, b.pt, v.pt, p.pt) == 12);
  CHECK(-tf::exact::orient3d_value<insphere_i32>(a.pt, b.pt, c.pt, p.pt) == 0);
  CHECK(+tf::exact::orient3d_value<insphere_i32>(a.pt, b.pt, c.pt, v.pt) ==
        -12);

  const std::array<insphere_site32, 5> upper = {a, b, c, u, p};
  const std::array<insphere_site32, 5> lower = {a, b, c, v, p};
  CHECK(tf::exact::insphere_sos(upper) == 1);
  CHECK(tf::exact::insphere_sos(lower) == -1);
  CHECK(tf::exact::insphere_conflict(upper, 1));
  CHECK(tf::exact::insphere_conflict(lower, -1));

  // The lower cell stored positively oriented, b and c exchanged, states
  // the same verdict.
  const std::array<insphere_site32, 5> lower_positive = {a, c, b, v, p};
  REQUIRE(tf::exact::orient3d_value<insphere_i32>(a.pt, c.pt, b.pt, v.pt) ==
          12);
  CHECK(tf::exact::insphere_conflict(lower_positive, 1));

  // And each agrees with the independently expanded perturbed determinant.
  const std::array<insphere_pt32, 5> upper_pts = {a.pt, b.pt, c.pt, u.pt, p.pt};
  const std::array<insphere_pt32, 5> lower_pts = {a.pt, b.pt, c.pt, v.pt, p.pt};
  const std::array<int, 5> ranks = {0, 1, 2, 3, 5};
  CHECK(tf::exact::insphere_sos(upper) ==
        insphere_perturbed_sign(upper_pts, ranks));
  CHECK(tf::exact::insphere_sos(lower) ==
        insphere_perturbed_sign(lower_pts, ranks));
}

TEST_CASE("insphere: filtered signs agree with the full integer determinant",
          "[exact][insphere][filter]") {
  std::mt19937_64 generator(9302026);
  std::size_t certified = 0, undecided = 0;
  for (unsigned bits : {1u, 8u, 30u, 37u, 40u, 50u, 52u, 60u, 63u}) {
    const auto span = (std::int64_t(1) << (bits - 1)) - 1;
    std::uniform_int_distribution<std::int64_t> coordinate(-span, span);
    for (std::size_t trial = 0; trial < 512; ++trial) {
      std::array<insphere_pt64, 5> points;
      for (auto &point : points)
        point = insphere_pt64{coordinate(generator), coordinate(generator),
                              coordinate(generator)};
      const auto value = tf::exact::insphere_value_scaled<insphere_i64>(
          points[0], points[1], points[2], points[3], points[4]);
      const int expected = value > 0 ? 1 : value < 0 ? -1 : 0;
      const int sign = tf::exact::insphere::filter_sign<insphere_i64>(
          points[0], points[1], points[2], points[3], points[4]);
      certified += sign != 0;
      undecided += sign == 0;
      CAPTURE(bits, trial);
      REQUIRE((sign == 0 || sign == expected));
      if (bits <= 40) {
        const int narrow = tf::exact::insphere::filter_sign<insphere_i32>(
            points[0], points[1], points[2], points[3], points[4]);
        const auto exact = tf::exact::insphere_value_scaled<insphere_i32>(
            points[0], points[1], points[2], points[3], points[4]);
        CHECK(narrow == sign);
        CHECK((exact > 0 ? 1 : exact < 0 ? -1 : 0) == expected);
      }
    }
  }
  CHECK(certified > 1000);
  CHECK(undecided > 1000);
}

TEST_CASE("insphere: filtering translates before conversion and defers ties",
          "[exact][insphere][filter]") {
  const insphere_i64 offset = insphere_i64(1) << 62;
  const std::array<insphere_pt64, 5> local{
      {{0, 0, 0}, {4, 0, 0}, {0, 4, 0}, {0, 0, 4}, {1, 1, 1}}};
  auto translated = local;
  for (auto &point : translated)
    for (std::size_t axis = 0; axis < 3; ++axis)
      point[axis] += offset;
  const int expected = tf::exact::insphere_sign<insphere_i64>(
      local[0], local[1], local[2], local[3], local[4]);
  REQUIRE(expected != 0);
  CHECK(tf::exact::insphere::filter_sign<insphere_i64>(
            translated[0], translated[1], translated[2], translated[3],
            translated[4]) == expected);
  for (unsigned bits : {10u, 30u, 50u, 60u}) {
    const insphere_i64 length = insphere_i64(1) << bits;
    const insphere_pt64 a{0, 0, 0}, b{length, 0, 0}, c{0, length, 0},
        d{0, 0, length};
    for (int delta : {-1, 0, 1}) {
      const insphere_pt64 query{length, length, length + delta};
      const auto value =
          tf::exact::insphere_value<insphere_i64>(a, b, c, d, query);
      const int exact = value > 0 ? 1 : value < 0 ? -1 : 0;
      const int filtered =
          tf::exact::insphere::filter_sign<insphere_i64>(a, b, c, d, query);
      CAPTURE(bits, delta);
      CHECK((filtered == 0 || filtered == exact));
      if (delta == 0 || bits >= 50)
        CHECK(filtered == 0);
    }
  }
}

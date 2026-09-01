/**
 * @file test_det2_sign.cpp
 * @brief tf::exact::det2_sign — exact sign of a*b - c*d past the ladder.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/det2_sign.hpp>
#include <trueform/exact/int256.hpp>
#include <trueform/exact/meta.hpp>

#include <array>
#include <cstdint>
#include <random>

using tf::exact::int256;
using tf::exact::int64;

namespace {

auto det2_pow2(unsigned e) -> int256 { return int256(1) << e; }

// Independent oracle: sign-magnitude schoolbook multiply on 32-bit
// limbs (portable, no __int128), lexicographic magnitude compare.
template <typename T, int Limbs>
auto to_limbs(T v) -> std::array<std::uint32_t, Limbs> {
  std::array<std::uint32_t, Limbs> out{};
  for (int k = 0; k < Limbs; ++k)
    out[std::size_t(k)] = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(v >> unsigned(32 * k)));
  return out;
}

template <int Limbs>
auto mul_limbs(const std::array<std::uint32_t, Limbs> &x,
               const std::array<std::uint32_t, Limbs> &y)
    -> std::array<std::uint32_t, 2 * Limbs> {
  std::array<std::uint32_t, 2 * Limbs> r{};
  for (int i = 0; i < Limbs; ++i) {
    std::uint64_t carry = 0;
    for (int j = 0; j < Limbs; ++j) {
      const std::uint64_t cur = std::uint64_t(x[std::size_t(i)]) *
                                    y[std::size_t(j)] +
                                r[std::size_t(i + j)] + carry;
      r[std::size_t(i + j)] = std::uint32_t(cur);
      carry = cur >> 32;
    }
    r[std::size_t(i + Limbs)] = std::uint32_t(carry);
  }
  return r;
}

template <typename T, int Limbs>
auto oracle_sign(T a, T b, T c, T d) -> int {
  const int sa = (a > T(0)) ? 1 : (a < T(0)) ? -1 : 0;
  const int sb = (b > T(0)) ? 1 : (b < T(0)) ? -1 : 0;
  const int sc = (c > T(0)) ? 1 : (c < T(0)) ? -1 : 0;
  const int sd = (d > T(0)) ? 1 : (d < T(0)) ? -1 : 0;
  const int s_ab = sa * sb;
  const int s_cd = sc * sd;
  const auto mag_ab = mul_limbs<Limbs>(to_limbs<T, Limbs>(sa < 0 ? -a : a),
                                       to_limbs<T, Limbs>(sb < 0 ? -b : b));
  const auto mag_cd = mul_limbs<Limbs>(to_limbs<T, Limbs>(sc < 0 ? -c : c),
                                       to_limbs<T, Limbs>(sd < 0 ? -d : d));
  if (s_ab != s_cd)
    return (s_ab > s_cd) ? 1 : -1;
  for (int k = 2 * Limbs - 1; k >= 0; --k) {
    if (mag_ab[std::size_t(k)] != mag_cd[std::size_t(k)])
      return ((mag_ab[std::size_t(k)] > mag_cd[std::size_t(k)]) == (s_ab > 0))
                 ? 1
                 : -1;
  }
  return 0;
}

auto random_t2(std::mt19937_64 &rng, unsigned bits) -> int256 {
  int256 v(0);
  for (unsigned k = 0; k < (bits + 63) / 64; ++k)
    v = (v << 64) | int256(std::uint64_t(rng()));
  v = v & ((int256(1) << bits) - int256(1));
  return (rng() & 1) ? -v : v;
}

} // namespace

TEST_CASE("det2_sign: in-range products match plain arithmetic",
          "[exact][det2_sign]") {
  REQUIRE(tf::exact::det2_sign<int64>(int256(3), int256(5), int256(2),
                                      int256(7)) == 1); // 15 - 14
  REQUIRE(tf::exact::det2_sign<int64>(int256(2), int256(7), int256(3),
                                      int256(5)) == -1);
  REQUIRE(tf::exact::det2_sign<int64>(int256(6), int256(4), int256(-8),
                                      int256(-3)) == 0); // 24 - 24
  REQUIRE(tf::exact::det2_sign<int64>(int256(0), int256(0), int256(0),
                                      int256(0)) == 0);
}

TEST_CASE("det2_sign: products past the type width keep the exact sign",
          "[exact][det2_sign]") {
  // 2^250 * 2^250 differs from (2^250 - 1) * (2^250 + 1) by exactly 1.
  REQUIRE(tf::exact::det2_sign<int64>(det2_pow2(250), det2_pow2(250),
                                      det2_pow2(250) - int256(1),
                                      det2_pow2(250) + int256(1)) == 1);
  REQUIRE(tf::exact::det2_sign<int64>(det2_pow2(250) - int256(1),
                                      det2_pow2(250) + int256(1),
                                      det2_pow2(250), det2_pow2(250)) == -1);
  // exact cancellation at full magnitude, mixed signs
  REQUIRE(tf::exact::det2_sign<int64>(det2_pow2(250), det2_pow2(249),
                                      -det2_pow2(249), -det2_pow2(250)) == 0);
  // -2^500 vs -2^250: the difference is negative.
  REQUIRE(tf::exact::det2_sign<int64>(-det2_pow2(250), det2_pow2(250),
                                      det2_pow2(125), -det2_pow2(125)) == -1);
}

TEST_CASE("det2_sign: randomized oracle sweep on the int64 ladder",
          "[exact][det2_sign]") {
  std::mt19937_64 rng(0x7f4a2b11u);
  for (int it = 0; it < 4000; ++it) {
    // Degree-3 operands (EF parameter nums/dens) top out near 193 bits;
    // sweep beyond, to the full T2 range.
    const unsigned bits = 32 + unsigned(rng() % 220);
    const auto a = random_t2(rng, bits);
    const auto b = random_t2(rng, bits);
    const auto c = random_t2(rng, bits);
    const auto d = random_t2(rng, bits);
    REQUIRE(tf::exact::det2_sign<int64>(a, b, c, d) ==
            (oracle_sign<int256, 8>(a, b, c, d)));
  }
}

TEST_CASE("det2_sign: near-tie products differ in the last bit",
          "[exact][det2_sign]") {
  std::mt19937_64 rng(0x2c9d1e05u);
  for (int it = 0; it < 2000; ++it) {
    const auto a = random_t2(rng, 200);
    const auto b = random_t2(rng, 50);
    // c*d == a*b exactly, then nudge one factor by one.
    const auto c = a;
    const auto d = b;
    REQUIRE(tf::exact::det2_sign<int64>(a, b, c, d) == 0);
    REQUIRE(tf::exact::det2_sign<int64>(a, b, c, d + int256(1)) ==
            oracle_sign<int256, 8>(a, b, c, d + int256(1)));
    REQUIRE(tf::exact::det2_sign<int64>(a + int256(1), b, c, d) ==
            oracle_sign<int256, 8>(a + int256(1), b, c, d));
  }
}

TEST_CASE("det2_sign: the radial turn between two lattice-extreme wedges",
          "[exact][det2_sign]") {
  using T1 = typename tf::exact::meta<tf::exact::int32>::T1;
  using T2 = typename tf::exact::meta<tf::exact::int32>::T2;
  // An integer-coordinate mesh is not rescaled, so its edge vectors reach
  // 2^32 and a face's exact normal reaches 2^65. The turn between two such
  // normals is one component of their cross — degree four, past every rung
  // the ladder has.
  const T1 e = T1(4294967294LL);
  const auto normal = [](const std::array<T1, 3> &x,
                         const std::array<T1, 3> &y) {
    return std::array<T2, 3>{T2(x[1]) * y[2] - T2(x[2]) * y[1],
                             T2(x[2]) * y[0] - T2(x[0]) * y[2],
                             T2(x[0]) * y[1] - T2(x[1]) * y[0]};
  };
  const auto a = normal({e, e, e}, {e, e, -e});
  const auto b = normal({e, -e, e}, {-e, e, e});
  const int256 turn =
      int256(a[0]) * int256(b[1]) - int256(a[1]) * int256(b[0]);
  REQUIRE(turn > (int256(1) << 127));
  REQUIRE(tf::exact::det2_sign<tf::exact::int32>(a[0], b[1], a[1], b[0]) == 1);
  REQUIRE(tf::exact::det2_sign<tf::exact::int32>(b[0], a[1], b[1], a[0]) == -1);
}

TEST_CASE("det2_sign: int32 ladder (T2 = int128) matches the oracle",
          "[exact][det2_sign]") {
  using T2 = typename tf::exact::meta<tf::exact::int32>::T2;
  std::mt19937_64 rng(0x51e0aa37u);
  for (int it = 0; it < 4000; ++it) {
    const unsigned bits = 16 + unsigned(rng() % 110);
    const auto rnd = [&](unsigned nbits) -> T2 {
      T2 v = T2(std::uint64_t(rng()));
      v = (v << 64) | T2(std::uint64_t(rng()));
      v = v & ((T2(1) << nbits) - T2(1));
      return (rng() & 1) ? T2(0) - v : v;
    };
    const auto a = rnd(bits), b = rnd(bits), c = rnd(bits), d = rnd(bits);
    REQUIRE(tf::exact::det2_sign<tf::exact::int32>(a, b, c, d) ==
            (oracle_sign<T2, 4>(a, b, c, d)));
  }
}

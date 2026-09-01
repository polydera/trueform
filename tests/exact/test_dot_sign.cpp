/**
 * @file test_dot_sign.cpp
 * @brief tf::exact::dot_sign — exact sign of overflowing dot products.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/dot_sign.hpp>
#include <trueform/exact/int256.hpp>

#include <random>

using tf::exact::int256;

namespace {
auto dot_pow2(unsigned e) -> int256 { return int256(1) << e; }
} // namespace

TEST_CASE("dot_sign: in-range products match plain arithmetic",
          "[exact][dot_sign]") {
  std::array<int256, 3> a{int256(3), int256(-5), int256(7)};
  std::array<int256, 3> b{int256(2), int256(4), int256(1)};
  // 6 - 20 + 7 = -7
  REQUIRE(tf::exact::dot_sign(a, b) == -1);
  std::array<int256, 3> c{int256(0), int256(0), int256(0)};
  REQUIRE(tf::exact::dot_sign(a, c) == 0);
}

TEST_CASE("dot_sign: products past the type width keep the exact sign",
          "[exact][dot_sign]") {
  // a . b = 2^250 * 2^120 - (2^250 - 1) * 2^120 = 2^120 > 0, but each
  // term wraps mod 2^256 under plain multiplication.
  std::array<int256, 3> a{dot_pow2(250), -(dot_pow2(250) - int256(1)),
                          int256(0)};
  std::array<int256, 3> b{dot_pow2(120), dot_pow2(120), int256(0)};
  REQUIRE(tf::exact::dot_sign(a, b) == 1);
  // flip one operand: exact value -2^120
  std::array<int256, 3> a2{-(dot_pow2(250)), dot_pow2(250) - int256(1),
                           int256(0)};
  REQUIRE(tf::exact::dot_sign(a2, b) == -1);
  // exact cancellation across huge terms
  std::array<int256, 3> a3{dot_pow2(250), -dot_pow2(250), int256(0)};
  REQUIRE(tf::exact::dot_sign(a3, b) == 0);
}

TEST_CASE("dot_sign: attainable extremes of the arrangement carrier",
          "[exact][dot_sign]") {
  using T = tf::exact::int128;
  // the component-maximal configuration under the 0.495-max converter:
  // two diagonal-rectangle wedges and their carrier (0.96 of the type
  // range — past the old 2^(2H-2) bound)
  const T m{std::int64_t(-9039826920395232200LL)};
  const T p{std::int64_t(9039826920395232200LL)};
  std::array<T, 3> n1{m, m, T(0)};
  std::array<T, 3> n2{p, m, T(0)};
  auto cross = [](const std::array<T, 3> &a, const std::array<T, 3> &b) {
    return std::array<T, 3>{a[1] * b[2] - a[2] * b[1],
                            a[2] * b[0] - a[0] * b[2],
                            a[0] * b[1] - a[1] * b[0]};
  };
  auto d = cross(n1, n2);

  // the carrier is exactly perpendicular to its defining wedges
  CHECK(tf::exact::dot_sign(d, n1) == 0);
  CHECK(tf::exact::dot_sign(d, n2) == 0);

  // a perturbed wedge tips the sign
  std::array<T, 3> n3{n1[0] + T(1), n1[1], n1[2] + T(12345)};
  CHECK(tf::exact::dot_sign(d, n3) == 1);
  std::array<T, 3> n4{n1[0] + T(1), n1[1], n1[2] - T(12345)};
  CHECK(tf::exact::dot_sign(d, n4) == -1);
}

TEST_CASE("dot_sign: three-term accumulation past the old bounds",
          "[exact][dot_sign]") {
  using T = tf::exact::int128;
  // |a| ~ 0.75 * 2^127, |b| ~ 0.75 * 2^63: every partial sum of the
  // naive split overflows; the renormalized accumulation must not
  const T amax = (T(1) << 126) + (T(1) << 125);
  const T bmax = (T(1) << 62) + (T(1) << 61);
  std::array<T, 3> a{amax, amax, amax};
  std::array<T, 3> b{bmax, bmax, bmax};
  CHECK(tf::exact::dot_sign(a, b) == 1);
  std::array<T, 3> bn{T(0) - bmax, T(0) - bmax, T(0) - bmax};
  CHECK(tf::exact::dot_sign(a, bn) == -1);
  // massive cancellation, tiny residue decides
  std::array<T, 3> ac{amax, T(0) - amax, T(3)};
  std::array<T, 3> bc{bmax, bmax, T(1)};
  CHECK(tf::exact::dot_sign(ac, bc) == 1);
  std::array<T, 3> ac2{amax, T(0) - amax, T(0) - T(3)};
  CHECK(tf::exact::dot_sign(ac2, bc) == -1);
  std::array<T, 3> az{amax, T(0) - amax, T(0)};
  CHECK(tf::exact::dot_sign(az, bc) == 0);
}

TEST_CASE("dot_sign: randomized differential vs promoted arithmetic",
          "[exact][dot_sign]") {
  using T = tf::exact::int128;
  using W = tf::exact::int256;
  std::mt19937_64 rng(20260715);
  auto rand_bits = [&](int bits) -> T {
    T v(0);
    for (int k = 0; k < bits; k += 32)
      v = (v << 32) + T(std::int64_t(rng() & 0xffffffffULL));
    return v >> (((bits + 31) / 32) * 32 - bits);
  };
  for (int it = 0; it < 2000; ++it) {
    std::array<T, 3> a, b;
    for (int c = 0; c < 3; ++c) {
      a[c] = rand_bits(126);
      if (rng() & 1)
        a[c] = T(0) - a[c];
      b[c] = rand_bits(62);
      if (rng() & 1)
        b[c] = T(0) - b[c];
    }
    W ref(0);
    for (int c = 0; c < 3; ++c)
      ref = ref + W(a[c]) * W(b[c]);
    int expected = ref > W(0) ? 1 : (ref < W(0) ? -1 : 0);
    REQUIRE(tf::exact::dot_sign(a, b) == expected);
  }
}

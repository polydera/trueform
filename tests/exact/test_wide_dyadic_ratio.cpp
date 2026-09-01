/**
 * @file test_wide_dyadic_ratio.cpp
 * @brief tf::exact::wide_dyadic_ratio — dyadic conversion past the
 *        narrow budget.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/dyadic_ratio.hpp>
#include <trueform/exact/int256.hpp>
#include <trueform/exact/wide_dyadic_ratio.hpp>

#include <random>

using tf::exact::int256;
using tf::exact::int64;

TEST_CASE("wide_dyadic_ratio: agrees with dyadic_ratio inside its budget",
          "[exact][wide_dyadic_ratio]") {
  std::mt19937_64 rng(0x3d1f9b21u);
  for (int it = 0; it < 2000; ++it) {
    const auto den = int256(std::uint64_t(rng()) | 1u);
    const auto num = int256(std::uint64_t(rng()) % std::uint64_t(rng() | 1u));
    REQUIRE(tf::exact::wide_dyadic_ratio<int64>(num, den) ==
            tf::exact::dyadic_ratio<int64>(num, den));
  }
  REQUIRE(tf::exact::wide_dyadic_ratio<int64>(int256(0), int256(7)) == 0);
  REQUIRE(tf::exact::wide_dyadic_ratio<int64>(int256(7), int256(7)) ==
          tf::exact::dyadic_ratio<int64>(int256(7), int256(7)));
}

TEST_CASE("wide_dyadic_ratio: degree-3 fractions keep the exact rounding",
          "[exact][wide_dyadic_ratio]") {
  // num/den ~190 bits — the edge-plane fraction width dyadic_ratio's
  // scale multiplication overflows on. Ground truth by construction:
  // num = den * k / 2^s exactly for dyadic k/2^s.
  const auto den = (int256(1) << 190) + (int256(1) << 100) + int256(12345);
  const auto bits = tf::exact::meta<int64>::param_bits;
  for (int s = 1; s <= 12; ++s)
    for (int k = 1; k < (1 << s); k += (1 << s) / 4 + 1) {
      const auto num = (den >> unsigned(s)) * int256(k);
      const auto expect =
          (typename tf::exact::meta<int64>::param_type(k))
          << unsigned(bits - s);
      const auto got = tf::exact::wide_dyadic_ratio<int64>(num, den);
      // num truncates den/2^s, so the true ratio sits within one grid
      // step below k/2^s.
      REQUIRE(got <= expect);
      REQUIRE(expect - got <= 1);
    }
  // the half-way rounding: num = den/2 exactly (den even)
  const auto even = int256(1) << 191;
  REQUIRE(tf::exact::wide_dyadic_ratio<int64>(even >> 1, even) ==
          (typename tf::exact::meta<int64>::param_type(1))
              << unsigned(bits - 1));
}

TEST_CASE("wide_dyadic_ratio: the narrow path provably overflows here",
          "[exact][wide_dyadic_ratio]") {
  // The witness: a 186-bit numerator sends dyadic_ratio's scale product
  // past T2 and the parameter negative.
  const auto den = (int256(1) << 189) + int256(3);
  const auto num = (int256(1) << 186) + int256(1);
  const auto narrow = tf::exact::dyadic_ratio<int64>(num, den);
  const auto wide = tf::exact::wide_dyadic_ratio<int64>(num, den);
  // The scale product wraps mod 2^256: whatever it yields is not the
  // value; the chunked division is.
  REQUIRE(narrow != wide);
  REQUIRE(wide > 0);
  REQUIRE(wide <= (typename tf::exact::meta<int64>::param_type(1))
                      << unsigned(tf::exact::meta<int64>::param_bits));
}

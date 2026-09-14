/**
 * @file test_int512.cpp
 * @brief Tests for tf::exact::int512
 *
 * int512 is past the meta ladder and has one consumer: the product rung the
 * int64 lattice's door stands on. What that consumer asks of it is what is
 * asserted here — the sign of a comparison, an exact product, an exact
 * quotient, and a shift that loses no bit.
 *
 * The ground truth is int256 wherever a value fits in one, and the algebra
 * itself where it does not: a reconstruction oracle for divmod over a width
 * sweep, and the identities a ring owes. The boundary values are the ones
 * whose magnitude carries the sign bit — INT512_MIN, whose magnitude is
 * 2^511 — because that is where a signed shift of a magnitude turns a
 * division into a walk.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <random>
#include <trueform/exact/int128.hpp>
#include <trueform/exact/int256.hpp>
#include <trueform/exact/int512.hpp>

using I128 = tf::exact::int128;
using U128 = tf::exact::uint128;
using I256 = tf::exact::int256;
using I512 = tf::exact::int512;

static auto int512_of_words(std::uint64_t x0, std::uint64_t x1,
                            std::uint64_t x2, std::uint64_t x3,
                            std::uint64_t x4, std::uint64_t x5,
                            std::uint64_t x6, std::uint64_t x7) -> I512 {
  return I512(U128(x0) | (U128(x1) << 64), U128(x2) | (U128(x3) << 64),
              U128(x4) | (U128(x5) << 64), U128(x6) | (U128(x7) << 64));
}

/// The most negative value, whose magnitude 2^511 is the one that carries
/// the sign bit.
static auto int512_min() -> I512 {
  return int512_of_words(0, 0, 0, 0, 0, 0, 0, 0x8000000000000000ull);
}

static auto int512_max() -> I512 {
  return int512_of_words(~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull,
                         0x7fffffffffffffffull);
}

static auto int512_magnitude(const I512 &v) -> I512 {
  return v.is_negative() ? -v : v;
}

/// 2^200, built through int256 because int128 cannot hold the shift.
static auto int512_two_pow_200() -> I256 {
  return I256(I128(1) << 100) * I256(I128(1) << 100);
}

namespace {

/// A deterministic value of at most the requested magnitude width.
struct int512_stream {
  std::mt19937_64 rng{20260909};

  auto of_bits(unsigned bits) -> I512 {
    I512 v(0);
    for (unsigned produced = 0; produced < bits; produced += 32) {
      const unsigned c = bits - produced < 32 ? bits - produced : 32;
      const auto chunk = rng() & ((std::uint64_t(1) << c) - 1);
      v = (v << c) + I512(chunk);
    }
    return v;
  }
};

} // namespace

// ============================================================================
// Construction and conversion — int256 is the ground truth where it reaches
// ============================================================================

TEST_CASE("int512 carries every int256 through unchanged", "[int512]") {
  const I256 values[] = {
      I256(0),
      I256(1),
      I256(-1),
      I256(INT64_MIN),
      I256(I128(1) << 100),
      I256(-(I128(1) << 100)),
      int512_two_pow_200(),
      -int512_two_pow_200(),
  };
  for (const auto &v : values) {
    const I512 w(v);
    REQUIRE(w.is_negative() == v.is_negative());
    REQUIRE(w.is_zero() == v.is_zero());
    REQUIRE(static_cast<I256>(w) == v);
  }
}

TEST_CASE("int512 sign-extends a narrow negative into every limb", "[int512]") {
  const I512 minus_one(-1);
  for (std::size_t k = 0; k < 4; ++k)
    REQUIRE(minus_one.limb(k) == ~U128(0));
  REQUIRE(minus_one.is_negative());

  const I512 from_i128(-(I128(1) << 100));
  REQUIRE(from_i128.limb(1) == ~U128(0));
  REQUIRE(from_i128.limb(2) == ~U128(0));
  REQUIRE(from_i128.limb(3) == ~U128(0));
}

TEST_CASE("int512 explicit conversions truncate to the low limbs", "[int512]") {
  const auto v = int512_of_words(0xdeadbeefcafebabeull, 0x0123456789abcdefull,
                                 2, 3, 4, 5, 6, 7);
  REQUIRE(static_cast<std::uint64_t>(v) == 0xdeadbeefcafebabeull);
  REQUIRE(static_cast<I128>(v) ==
          (I128(0xdeadbeefcafebabeull) | (I128(0x0123456789abcdefull) << 64)));
  REQUIRE(static_cast<I256>(v).lo() == v.limb(0));
  REQUIRE(static_cast<I256>(v).hi() == v.limb(1));
  REQUIRE(static_cast<bool>(v));
  REQUIRE(!static_cast<bool>(I512(0)));
}

// ============================================================================
// Comparison
// ============================================================================

TEST_CASE("int512 comparison agrees with int256", "[int512]") {
  const I256 values[] = {
      -int512_two_pow_200(), I256(-(I128(1) << 120)), I256(-1),
      I256(0),               I256(1),                 I256(I128(1) << 120),
      int512_two_pow_200(),
  };
  for (const auto &a : values)
    for (const auto &b : values) {
      const I512 wa(a), wb(b);
      REQUIRE((wa == wb) == (a == b));
      REQUIRE((wa != wb) == (a != b));
      REQUIRE((wa < wb) == (a < b));
      REQUIRE((wa > wb) == (a > b));
      REQUIRE((wa <= wb) == (a <= b));
      REQUIRE((wa >= wb) == (a >= b));
    }
}

TEST_CASE("int512 comparison orders the boundary values", "[int512]") {
  REQUIRE(int512_min().is_negative());
  REQUIRE(!int512_max().is_negative());
  REQUIRE(int512_min() < int512_max());
  REQUIRE(int512_min() < I512(0));
  REQUIRE(I512(0) < int512_max());
  REQUIRE(int512_min() < I512(-1));
  REQUIRE(int512_max() > I512(1));
  // the magnitude of the most negative value is itself
  REQUIRE(-int512_min() == int512_min());
  REQUIRE(int512_magnitude(int512_min()) == int512_min());
  REQUIRE(int512_min() + I512(1) == -int512_max());
  REQUIRE(int512_max() == (I512(1) << 511) - I512(1));
}

// ============================================================================
// Addition, subtraction, negation
// ============================================================================

TEST_CASE("int512 add and sub agree with int256 where both reach", "[int512]") {
  const I256 values[] = {
      I256(-(I128(1) << 100)), I256(-7),             I256(0),
      I256(7),                 I256(I128(1) << 100), I256(I128(1) << 120),
  };
  for (const auto &a : values)
    for (const auto &b : values) {
      REQUIRE(static_cast<I256>(I512(a) + I512(b)) == a + b);
      REQUIRE(static_cast<I256>(I512(a) - I512(b)) == a - b);
      REQUIRE(static_cast<I256>(-I512(a)) == -a);
    }
}

TEST_CASE("int512 add carries across every limb", "[int512]") {
  const auto below = int512_of_words(~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull,
                                     ~0ull, 0x7ffffffffffffffeull);
  REQUIRE(below + I512(1) ==
          int512_of_words(0, 0, 0, 0, 0, 0, 0, 0x7fffffffffffffffull));
  REQUIRE(int512_max() + I512(1) == int512_min());
  REQUIRE(I512(0) - int512_min() == int512_min());
}

// ============================================================================
// Multiplication
// ============================================================================

TEST_CASE("int512 multiply agrees with int256 where both reach", "[int512]") {
  const I256 values[] = {I256(-1),  I256(0),             I256(1),
                         I256(-7),  I256(13),            I256(I128(1) << 63),
                         I256(I128(1) << 100)};
  for (const auto &a : values)
    for (const auto &b : values)
      REQUIRE(static_cast<I256>(I512(a) * I512(b)) == a * b);
}

TEST_CASE("int512 widening multiply reaches the top limb", "[int512]") {
  REQUIRE((I512(1) << 200) * (I512(1) << 300) == (I512(1) << 500));
  REQUIRE((I512(1) << 255) * (I512(1) << 255) == (I512(1) << 510));
  REQUIRE((I512(1) << 256) * (I512(1) << 256) == I512(0));
}

TEST_CASE("int512 is a ring at full width", "[int512]") {
  int512_stream stream;
  for (int it = 0; it < 200; ++it) {
    const auto a = stream.of_bits(1 + unsigned(stream.rng() % 250));
    const auto b = stream.of_bits(1 + unsigned(stream.rng() % 250));
    const auto c = stream.of_bits(1 + unsigned(stream.rng() % 250));
    REQUIRE(a + b == b + a);
    REQUIRE(a * b == b * a);
    REQUIRE((a + b) - b == a);
    REQUIRE(a * (b + c) == a * b + a * c);
    REQUIRE((a + b) * (a - b) == a * a - b * b);
    REQUIRE(a + (-a) == I512(0));
    REQUIRE(a * I512(-1) == -a);
  }
}

// ============================================================================
// Division — the reconstruction oracle
// ============================================================================

TEST_CASE("int512 divmod: reconstruction oracle across widths and signs",
          "[int512]") {
  int512_stream stream;
  for (int it = 0; it < 20000; ++it) {
    const unsigned nb = 1 + unsigned(stream.rng() % 511);
    const unsigned db = 1 + unsigned(stream.rng() % nb);
    auto num = stream.of_bits(nb);
    auto den = stream.of_bits(db);
    if (den == I512(0))
      den = I512(1);
    if (stream.rng() & 1)
      num = -num;
    if (stream.rng() & 1)
      den = -den;
    const auto [q, r] = divmod(num, den);
    REQUIRE(q * den + r == num);
    // C++ semantics: the remainder carries the dividend's sign, |r| < |den|
    REQUIRE(int512_magnitude(r) < int512_magnitude(den));
    if (!r.is_zero())
      REQUIRE(r.is_negative() == num.is_negative());
  }
}

TEST_CASE("int512 divmod agrees with int256 where both reach", "[int512]") {
  const I256 dividends[] = {I256(0),   I256(1),   I256(-1),
                            I256(100), I256(-100), int512_two_pow_200(),
                            -int512_two_pow_200()};
  const I256 divisors[] = {I256(1),  I256(-1),            I256(7),
                           I256(-7), I256(I128(1) << 32), I256(I128(1) << 80)};
  for (const auto &a : dividends)
    for (const auto &b : divisors) {
      const auto [q512, r512] = divmod(I512(a), I512(b));
      const auto [q256, r256] = divmod(a, b);
      REQUIRE(static_cast<I256>(q512) == q256);
      REQUIRE(static_cast<I256>(r512) == r256);
    }
}

TEST_CASE("int512 divmod answers the most negative value", "[int512]") {
  // 2^511 is the one magnitude whose top bit is the sign bit. Shifted as a
  // signed value it reads as -1, every quotient-digit estimate collapses,
  // and the exact-correction loop is left walking the quotient one divisor
  // at a time — so these cases answer at all only because the magnitude is
  // shifted logically.
  const auto min = int512_min();

  const auto [q_pow, r_pow] = divmod(min, I512(1) << 100);
  REQUIRE(r_pow.is_zero());
  REQUIRE(q_pow == -(I512(1) << 411));
  REQUIRE(q_pow * (I512(1) << 100) + r_pow == min);

  // the narrow-divisor path
  const auto [q_narrow, r_narrow] = divmod(min, I512(7));
  REQUIRE(q_narrow * I512(7) + r_narrow == min);
  REQUIRE(int512_magnitude(r_narrow) < I512(7));
  REQUIRE(r_narrow.is_negative());

  // the wide-divisor path, whose estimate reads the shifted magnitude
  const auto den = int512_of_words(1, 0, 0, 0x1234567800000000ull, 0, 0, 0, 0);
  const auto [q_wide, r_wide] = divmod(min, den);
  REQUIRE(q_wide * den + r_wide == min);
  REQUIRE(int512_magnitude(r_wide) < int512_magnitude(den));

  REQUIRE(divmod(min, min).first == I512(1));
  REQUIRE(divmod(min, min).second.is_zero());
  REQUIRE(divmod(min, I512(1)).first == min);
  REQUIRE(divmod(min, I512(2)).first == -(I512(1) << 510));
  // the wrap the type shares with every two's-complement division
  REQUIRE(divmod(min, I512(-1)).first == min);

  const auto [q_near, r_near] = divmod(min + I512(1), I512(1) << 100);
  REQUIRE(q_near == -((I512(1) << 411) - I512(1)));
  REQUIRE(q_near * (I512(1) << 100) + r_near == min + I512(1));
}

// ============================================================================
// Shifts
// ============================================================================

TEST_CASE("int512 shift round-trips at every width", "[int512]") {
  int512_stream stream;
  for (int it = 0; it < 60; ++it) {
    const auto v = stream.of_bits(1 + unsigned(stream.rng() % 200));
    const auto negative = -v;
    for (unsigned s = 0; s < 300; ++s) {
      REQUIRE(((v << s) >> s) == v);
      REQUIRE(((negative << s) >> s) == negative);
    }
  }
}

TEST_CASE("int512 shift crosses every limb boundary", "[int512]") {
  const I512 one(1);
  for (unsigned s = 0; s < 512; ++s) {
    const auto shifted = one << s;
    REQUIRE(!shifted.is_zero());
    REQUIRE(shifted.limb(s / 128) == (U128(1) << (s % 128)));
    REQUIRE(shifted.is_negative() == (s == 511));
    // the top bit is the sign, so the arithmetic shift brings back -1 there
    // and the bit everywhere below it
    REQUIRE((shifted >> s) == (s == 511 ? I512(-1) : one));
  }
  REQUIRE((one << 512) == I512(0));
  REQUIRE((I512(-1) >> 512) == I512(-1));
  REQUIRE((one >> 512) == I512(0));
}

TEST_CASE("int512 right shift is arithmetic", "[int512]") {
  REQUIRE((I512(-100) >> 1) == I512(-50));
  REQUIRE((int512_min() >> 511) == I512(-1));
  REQUIRE((int512_min() >> 1) == -(I512(1) << 510));
  REQUIRE((int512_max() >> 510) == I512(1));
}

// ============================================================================
// Bitwise and compound assignment
// ============================================================================

TEST_CASE("int512 bitwise identities hold at full width", "[int512]") {
  const auto a = int512_of_words(0x1234, 0x5678, 0x9abc, 0xdef0, 0x0f0f,
                                 0xf0f0, 0xaaaa, 0x5555);
  REQUIRE(~~a == a);
  REQUIRE((a & ~a) == I512(0));
  REQUIRE((a | ~a) == I512(-1));
  REQUIRE((a ^ a) == I512(0));
  REQUIRE((a ^ I512(0)) == a);
}

TEST_CASE("int512 compound assignment and stepping", "[int512]") {
  I512 a(100);
  a += I512(50);
  REQUIRE(a == I512(150));
  a -= I512(200);
  REQUIRE(a == I512(-50));
  a *= I512(-2);
  REQUIRE(a == I512(100));
  a /= I512(3);
  REQUIRE(a == I512(33));
  a %= I512(10);
  REQUIRE(a == I512(3));
  a <<= 300;
  a >>= 300;
  REQUIRE(a == I512(3));

  I512 b(0);
  REQUIRE(++b == I512(1));
  REQUIRE(b++ == I512(1));
  REQUIRE(b == I512(2));
  REQUIRE(--b == I512(1));
  REQUIRE(b-- == I512(1));
  REQUIRE(b == I512(0));
  --b;
  REQUIRE(b == I512(-1));
}

// ============================================================================
// The double read the screens use
// ============================================================================

TEST_CASE("int512 reads as a double without calling a high limb negative",
          "[int512]") {
  REQUIRE(static_cast<double>(I512(0)) == 0.0);
  REQUIRE(static_cast<double>(I512(1)) == 1.0);
  REQUIRE(static_cast<double>(I512(-1)) == -1.0);
  // a value whose low half alone reads negative as an int256
  const auto low_top = I512(U128(0), U128(1) << 127, U128(0), U128(0));
  REQUIRE(static_cast<double>(low_top) > 0.0);
  REQUIRE(static_cast<double>(-low_top) < 0.0);
  REQUIRE(static_cast<double>(low_top) == -static_cast<double>(-low_top));
  REQUIRE(static_cast<double>(int512_min()) < 0.0);
  REQUIRE(static_cast<double>(int512_max()) > 0.0);
}

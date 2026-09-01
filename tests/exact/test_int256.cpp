/**
 * @file test_int256.cpp
 * @brief Tests for tf::exact::int256
 *
 * Covers construction, comparison, arithmetic, shifts, bitwise,
 * conversions, and large-value cross-checks. All hardcoded results
 * verified against boost::multiprecision.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <random>
#include <trueform/exact/int128.hpp>
#include <trueform/exact/int256.hpp>

using I128 = tf::exact::int128;
using U128 = tf::exact::uint128;
using I256 = tf::exact::int256;

static auto to_i128(const I256 &v) -> I128 {
  return static_cast<I128>(v.lo());
}

static auto make256(uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3)
    -> I256 {
  U128 lo = U128(x0) | (U128(x1) << 64);
  U128 hi = U128(x2) | (U128(x3) << 64);
  return I256(lo, hi);
}

// ============================================================================
// Construction
// ============================================================================

TEST_CASE("int256 from zero", "[int256]") {
  I256 z(0);
  REQUIRE(z.is_zero());
  REQUIRE(!z.is_negative());
}

TEST_CASE("int256 from positive int", "[int256]") {
  I256 p(42);
  REQUIRE(!p.is_zero());
  REQUIRE(!p.is_negative());
  REQUIRE(static_cast<int64_t>(p) == 42);
}

TEST_CASE("int256 from negative int", "[int256]") {
  I256 n(-1);
  REQUIRE(!n.is_zero());
  REQUIRE(n.is_negative());
  REQUIRE(static_cast<int64_t>(n) == -1);
  REQUIRE(n.lo() == ~U128(0));
  REQUIRE(n.hi() == ~U128(0));
}

TEST_CASE("int256 from INT64_MIN", "[int256]") {
  I256 mn(INT64_MIN);
  REQUIRE(mn.is_negative());
  REQUIRE(static_cast<int64_t>(mn) == INT64_MIN);
  REQUIRE(mn.hi() == ~U128(0));
}

TEST_CASE("int256 from UINT64_MAX", "[int256]") {
  I256 u(uint64_t(UINT64_MAX));
  REQUIRE(!u.is_negative());
  REQUIRE(u.lo() == U128(UINT64_MAX));
  REQUIRE(u.hi() == 0);
}

TEST_CASE("int256 from positive int128", "[int256]") {
  I128 v = I128(1) << 100;
  I256 w(v);
  REQUIRE(!w.is_negative());
  REQUIRE(to_i128(w) == v);
  REQUIRE(w.hi() == 0);
}

TEST_CASE("int256 from negative int128", "[int256]") {
  I128 v = -(I128(1) << 100);
  I256 w(v);
  REQUIRE(w.is_negative());
  REQUIRE(to_i128(w) == v);
  REQUIRE(w.hi() == ~U128(0));
}

TEST_CASE("int256 from 2 limbs", "[int256]") {
  U128 lo = U128(1) | (U128(2) << 64);
  U128 hi = U128(3) | (U128(4) << 64);
  I256 raw(lo, hi);
  REQUIRE(raw.lo() == lo);
  REQUIRE(raw.hi() == hi);
}

// ============================================================================
// Comparison — verified against int128 ground truth
// ============================================================================

TEST_CASE("int256 comparison", "[int256]") {
  I128 vals[] = {
      -(I128(1) << 120), -(I128(1) << 64), I128(-1000), I128(-1), I128(0),
      I128(1),           I128(1000),        I128(1) << 64, I128(1) << 120,
  };
  for (auto &vi : vals) {
    for (auto &vj : vals) {
      I256 a(vi), b(vj);
      REQUIRE((a == b) == (vi == vj));
      REQUIRE((a != b) == (vi != vj));
      REQUIRE((a < b) == (vi < vj));
      REQUIRE((a > b) == (vi > vj));
      REQUIRE((a <= b) == (vi <= vj));
      REQUIRE((a >= b) == (vi >= vj));
    }
  }
}

// ============================================================================
// Addition & subtraction — boost-verified hardcoded
// ============================================================================

TEST_CASE("int256 add/sub small values", "[int256]") {
  I128 vals[] = {
      -(I128(1) << 100), -(I128(1) << 64), I128(-7), I128(0),
      I128(7),           I128(1) << 64,     I128(1) << 100,
  };
  for (auto &vi : vals)
    for (auto &vj : vals) {
      I256 wa(vi), wb(vj);
      REQUIRE(to_i128(wa + wb) == vi + vj);
      REQUIRE(to_i128(wa - wb) == vi - vj);
      REQUIRE(to_i128(-wa) == -vi);
    }
}

TEST_CASE("int256 add overflow to 2^127", "[int256]") {
  I128 big = I128(1) << 126;
  I256 sum = I256(big) + I256(big);
  REQUIRE(!sum.is_negative());
  REQUIRE(sum.hi() == 0);
  REQUIRE(sum / I256(2) == I256(big));
}

TEST_CASE("int256 hardcoded add", "[int256]") {
  auto full_pos = make256(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                          0x1111111111111111ull, 0x2222222222222222ull);
  auto full_neg = make256(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                          0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);
  auto scattered = make256(0xAAAAAAAAAAAAAAAAull, 0x5555555555555555ull,
                           0xCCCCCCCCCCCCCCCCull, 0x3333333333333333ull);

  REQUIRE((full_pos + full_neg) == make256(
      0xf0e2156865bb99aeULL, 0xffffffffffffffffULL,
      0x0fedcba987654321ULL, 0xadcfed202dd01230ULL));

  REQUIRE((full_pos + scattered) == make256(
      0xbcdf01234567899aULL, 0x54320fedcba98765ULL,
      0xdddddddddddddddeULL, 0x5555555555555555ULL));

  REQUIRE((I256(uint64_t(UINT64_MAX)) + I256(uint64_t(UINT64_MAX))) == make256(
      0xfffffffffffffffeULL, 0x0000000000000001ULL,
      0x0000000000000000ULL, 0x0000000000000000ULL));
}

TEST_CASE("int256 hardcoded sub", "[int256]") {
  auto full_pos = make256(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                          0x1111111111111111ull, 0x2222222222222222ull);
  auto full_neg = make256(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                          0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);
  auto scattered = make256(0xAAAAAAAAAAAAAAAAull, 0x5555555555555555ull,
                           0xCCCCCCCCCCCCCCCCull, 0x3333333333333333ull);

  REQUIRE((I256(0) - full_neg) == make256(
      0x2152411035014542ULL, 0xfedcba9876543210ULL,
      0x0123456789abcdefULL, 0x74523501f4520ff2ULL));

  REQUIRE((I256(INT64_MIN) - I256(INT64_MAX)) == make256(
      0x0000000000000001ULL, 0xffffffffffffffffULL,
      0xffffffffffffffffULL, 0xffffffffffffffffULL));

  REQUIRE((full_pos - scattered) == make256(
      0x6789abcdf0123446ULL, 0xa987654320fedcbaULL,
      0x4444444444444445ULL, 0xeeeeeeeeeeeeeeeeULL));
}

// ============================================================================
// Negation — boost-verified
// ============================================================================

TEST_CASE("int256 hardcoded negate", "[int256]") {
  auto full_pos = make256(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                          0x1111111111111111ull, 0x2222222222222222ull);
  auto full_neg = make256(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                          0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);

  REQUIRE(-full_pos == make256(
      0xedcba98765432110ULL, 0x0123456789abcdefULL,
      0xeeeeeeeeeeeeeeeeULL, 0xddddddddddddddddULL));

  REQUIRE(-full_neg == make256(
      0x2152411035014542ULL, 0xfedcba9876543210ULL,
      0x0123456789abcdefULL, 0x74523501f4520ff2ULL));

  REQUIRE(-I256(1) == make256(
      0xffffffffffffffffULL, 0xffffffffffffffffULL,
      0xffffffffffffffffULL, 0xffffffffffffffffULL));
}

// ============================================================================
// Multiplication — boost-verified
// ============================================================================

TEST_CASE("int256 mul small values round-trip", "[int256]") {
  I128 vals[] = {I128(-1),  I128(0),           I128(1),
                 I128(-7),  I128(13),          I128(1) << 63,
                 I128(1) << 100, -(I128(1) << 100)};
  for (auto &vi : vals)
    for (auto &vj : vals) {
      I256 wa(vi), wb(vj);
      I256 product = wa * wb;
      if (vi != 0)
        REQUIRE(product / wa == wb);
      if (vj != 0)
        REQUIRE(product / wb == wa);
    }
}

TEST_CASE("int256 hardcoded mul", "[int256]") {
  auto full_pos = make256(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                          0x1111111111111111ull, 0x2222222222222222ull);
  auto full_neg = make256(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                          0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);
  auto scattered = make256(0xAAAAAAAAAAAAAAAAull, 0x5555555555555555ull,
                           0xCCCCCCCCCCCCCCCCull, 0x3333333333333333ull);
  auto max_pos = make256(0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull,
                         0xFFFFFFFFFFFFFFFFull, 0x7FFFFFFFFFFFFFFFull);

  REQUIRE(full_pos * full_neg == make256(
      0xeb689f4ea447d620ULL, 0x7ccf959345111bc7ULL,
      0x7af77f6344f4e5fbULL, 0xd36dbd20f751d578ULL));

  REQUIRE(I256(I128(1) << 100) * I256(-(I128(1) << 100)) == make256(
      0x0000000000000000ULL, 0x0000000000000000ULL,
      0x0000000000000000ULL, 0xffffffffffffff00ULL));

  REQUIRE(I256(INT64_MAX) * I256(INT64_MIN) == make256(
      0x8000000000000000ULL, 0xc000000000000000ULL,
      0xffffffffffffffffULL, 0xffffffffffffffffULL));

  REQUIRE(I256(uint64_t(UINT64_MAX)) * I256(uint64_t(UINT64_MAX)) == make256(
      0x0000000000000001ULL, 0xfffffffffffffffeULL,
      0x0000000000000000ULL, 0x0000000000000000ULL));

  REQUIRE(full_pos * scattered == make256(
      0xf3dd1baf98d76b60ULL, 0x5c28f5c28f5c28efULL,
      0xebbf5fcd070de189ULL, 0x2d96433469e3a199ULL));

  REQUIRE(max_pos * I256(-1) == make256(
      0x0000000000000001ULL, 0x0000000000000000ULL,
      0x0000000000000000ULL, 0x8000000000000000ULL));
}

TEST_CASE("int256 widening mul 2^100 * 2^100 = 2^200", "[int256]") {
  I128 a = I128(1) << 100;
  I256 product = I256(a) * I256(a);
  REQUIRE(!product.is_negative());
  REQUIRE(product.lo() == 0);
  REQUIRE(product.hi() == (U128(1) << 72));
}

TEST_CASE("int256 orient3d widening multiply", "[int256]") {
  I128 orient_a = (I128(1) << 100) + 12345;
  I128 orient_b = -(I128(1) << 95) + 67890;
  REQUIRE(I256(orient_a) * I256(orient_b) == make256(
      0x0000000031f46c22ULL, 0x00107b0380000000ULL,
      0x0000000000000000ULL, 0xfffffffffffffff8ULL));
}

// ============================================================================
// Multiply chains — boost-verified
// ============================================================================

TEST_CASE("int256 multiply accumulation", "[int256]") {
  auto full_pos = make256(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                          0x1111111111111111ull, 0x2222222222222222ull);
  auto full_neg = make256(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                          0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);
  auto scattered = make256(0xAAAAAAAAAAAAAAAAull, 0x5555555555555555ull,
                           0xCCCCCCCCCCCCCCCCull, 0x3333333333333333ull);
  auto max_pos = make256(0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull,
                         0xFFFFFFFFFFFFFFFFull, 0x7FFFFFFFFFFFFFFFull);

  REQUIRE((full_pos * full_neg + scattered * max_pos - full_pos * scattered) ==
          make256(0x4ce0d8f460c5c016ULL, 0xcb514a7b605f9d82ULL,
                  0xc26b52c9711a37a5ULL, 0x72a446b95a3b00aaULL));

  // difference of squares: (a+b)*(a-b) == a*a - b*b
  auto lhs = (full_pos + full_neg) * (full_pos - full_neg);
  auto rhs = full_pos * full_pos - full_neg * full_neg;
  REQUIRE(lhs == rhs);
  REQUIRE(lhs == make256(
      0xf34c55a201647bfcULL, 0x890797db890c55ffULL,
      0xdce644136e053007ULL, 0x4156a6a93f041fb4ULL));

  // distributivity
  REQUIRE(scattered * (full_pos + full_neg) ==
          scattered * full_pos + scattered * full_neg);
  REQUIRE(scattered * (full_pos + full_neg) == make256(
      0x5f69470fbc2d998cULL, 0xa5a0b1cd773e888fULL,
      0xa71a18e6cb982520ULL, 0x330807fb4ba13342ULL));
}

TEST_CASE("int256 powers", "[int256]") {
  // 7^30
  I256 base7(7), r7(1);
  for (int i = 0; i < 30; ++i)
    r7 = r7 * base7;
  REQUIRE(r7 == make256(
      0x15e1e1b36ff883d1ULL, 0x000000000012a4e4ULL,
      0x0000000000000000ULL, 0x0000000000000000ULL));

  // (-3)^40
  I256 base3(-3), r3(1);
  for (int i = 0; i < 40; ++i)
    r3 = r3 * base3;
  REQUIRE(r3 == make256(
      0xa8b8b452291fe821ULL, 0x0000000000000000ULL,
      0x0000000000000000ULL, 0x0000000000000000ULL));
}

// ============================================================================
// Division & modulo — boost-verified
// ============================================================================

TEST_CASE("int256 divmod algebraic identity", "[int256]") {
  I128 dividends[] = {I128(0),  I128(1),           I128(-1),
                      I128(100), I128(-100),        I128(1) << 100,
                      -(I128(1) << 100)};
  I128 divisors[] = {I128(1),  I128(-1), I128(7),
                     I128(-7), I128(1) << 32, I128(1) << 80};
  for (auto &a : dividends)
    for (auto &b : divisors) {
      auto [q, r] = divmod(I256(a), I256(b));
      REQUIRE(q * I256(b) + r == I256(a));
      if (!r.is_zero())
        REQUIRE(r.is_negative() == I256(a).is_negative());
    }
}

TEST_CASE("int256 hardcoded div", "[int256]") {
  auto full_pos = make256(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                          0x1111111111111111ull, 0x2222222222222222ull);
  auto full_neg = make256(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                          0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);
  auto scattered = make256(0xAAAAAAAAAAAAAAAAull, 0x5555555555555555ull,
                           0xCCCCCCCCCCCCCCCCull, 0x3333333333333333ull);

  REQUIRE(full_pos / I256(INT64_MAX) == make256(
      0x530eca8641fdb97aULL, 0xaaaaaaaaaaaaaaadULL,
      0x4444444444444444ULL, 0x0000000000000000ULL));
  REQUIRE(full_pos % I256(INT64_MAX) == make256(
      0x654320fedcba986aULL, 0x0000000000000000ULL,
      0x0000000000000000ULL, 0x0000000000000000ULL));

  REQUIRE(full_neg / I256(I128(1) << 100) == make256(
      0x8765432100123457ULL, 0xe0badf00dfedcba9ULL,
      0xfffffffff8badcafULL, 0xffffffffffffffffULL));
  REQUIRE(full_neg % I256(I128(1) << 100) == make256(
      0xdeadbeefcafebabeULL, 0xfffffff789abcdefULL,
      0xffffffffffffffffULL, 0xffffffffffffffffULL));

  REQUIRE(full_pos / full_neg == I256(0));
  REQUIRE(full_pos % full_neg == full_pos);

  REQUIRE(scattered / full_pos == make256(
      0x0000000000000001ULL, 0x0000000000000000ULL,
      0x0000000000000000ULL, 0x0000000000000000ULL));
  REQUIRE(scattered % full_pos == make256(
      0x987654320fedcbbaULL, 0x56789abcdf012345ULL,
      0xbbbbbbbbbbbbbbbaULL, 0x1111111111111111ULL));

  REQUIRE(full_neg / I256(-7) == make256(
      0x970bc026e3002e77ULL, 0x48faf615c7c2e294ULL,
      0x24bbe557ef188b22ULL, 0x109e0792909e0247ULL));
  REQUIRE(full_neg % I256(-7) == make256(
      0xffffffffffffffffULL, 0xffffffffffffffffULL,
      0xffffffffffffffffULL, 0xffffffffffffffffULL));
}

TEST_CASE("int256 large division", "[int256]") {
  auto num = I256(1) << 200;
  REQUIRE(num / I256(7) == make256(
      0x4924924924924924ULL, 0x2492492492492492ULL,
      0x9249249249249249ULL, 0x0000000000000024ULL));
  REQUIRE(num % I256(7) == make256(
      0x0000000000000004ULL, 0x0000000000000000ULL,
      0x0000000000000000ULL, 0x0000000000000000ULL));

  I128 big = I128(1) << 100;
  I256 big200 = I256(big) * I256(big);
  REQUIRE(big200 / I256(big) == I256(big));
}

TEST_CASE("int256 widening div round-trip", "[int256]") {
  I128 orient_a = (I128(1) << 100) + 12345;
  I128 orient_b = -(I128(1) << 95) + 67890;
  I256 num = I256(orient_a) * I256(orient_b);
  REQUIRE(num / I256(orient_a) == I256(orient_b));
}

// ============================================================================
// Shifts — boost-verified
// ============================================================================

TEST_CASE("int256 shift round-trip", "[int256]") {
  I128 vals[] = {I128(1), I128(-1), I128(0x123456789ABCDEF0ll),
                 I128(1) << 100, -(I128(1) << 100)};
  for (auto &v : vals) {
    I256 w(v);
    for (unsigned s = 0; s < 128; ++s)
      REQUIRE(((w << s) >> s) == w);
  }
}

TEST_CASE("int256 hardcoded shifts", "[int256]") {
  auto full_pos = make256(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                          0x1111111111111111ull, 0x2222222222222222ull);
  auto full_neg = make256(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                          0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);
  auto scattered = make256(0xAAAAAAAAAAAAAAAAull, 0x5555555555555555ull,
                           0xCCCCCCCCCCCCCCCCull, 0x3333333333333333ull);

  REQUIRE((full_pos << 1) == make256(
      0x2468acf13579bde0ULL, 0xfdb97530eca86420ULL,
      0x2222222222222223ULL, 0x4444444444444444ULL));

  REQUIRE((full_pos >> 1) == make256(
      0x091a2b3c4d5e6f78ULL, 0xff6e5d4c3b2a1908ULL,
      0x0888888888888888ULL, 0x1111111111111111ULL));

  REQUIRE((full_neg >> 3) == make256(
      0xfbd5b7ddf95fd757ULL, 0x002468acf13579bdULL,
      0xbfdb97530eca8642ULL, 0xf175b95fc175be01ULL));

  REQUIRE((I256(1) << 200) == make256(
      0x0000000000000000ULL, 0x0000000000000000ULL,
      0x0000000000000000ULL, 0x0000000000000100ULL));

  REQUIRE((I256(I128(1) << 100) << 50) == make256(
      0x0000000000000000ULL, 0x0000000000000000ULL,
      0x0000000000400000ULL, 0x0000000000000000ULL));

  REQUIRE((scattered << 65) == make256(
      0x0000000000000000ULL, 0x5555555555555554ULL,
      0xaaaaaaaaaaaaaaabULL, 0x9999999999999998ULL));

  REQUIRE((scattered >> 65) == make256(
      0x2aaaaaaaaaaaaaaaULL, 0xe666666666666666ULL,
      0x1999999999999999ULL, 0x0000000000000000ULL));
}

TEST_CASE("int256 shift at 128-bit boundary", "[int256]") {
  auto full_pos = make256(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                          0x1111111111111111ull, 0x2222222222222222ull);
  auto full_neg = make256(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                          0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);

  REQUIRE((full_pos << 127) == make256(
      0x0000000000000000ULL, 0x0000000000000000ULL,
      0x091a2b3c4d5e6f78ULL, 0xff6e5d4c3b2a1908ULL));

  REQUIRE((full_pos << 128) == make256(
      0x0000000000000000ULL, 0x0000000000000000ULL,
      0x123456789abcdef0ULL, 0xfedcba9876543210ULL));

  REQUIRE((full_pos << 129) == make256(
      0x0000000000000000ULL, 0x0000000000000000ULL,
      0x2468acf13579bde0ULL, 0xfdb97530eca86420ULL));

  REQUIRE((full_neg >> 127) == make256(
      0xfdb97530eca86420ULL, 0x175b95fc175be01bULL,
      0xffffffffffffffffULL, 0xffffffffffffffffULL));

  REQUIRE((full_neg >> 128) == make256(
      0xfedcba9876543210ULL, 0x8badcafe0badf00dULL,
      0xffffffffffffffffULL, 0xffffffffffffffffULL));

  REQUIRE((full_neg >> 129) == make256(
      0xff6e5d4c3b2a1908ULL, 0xc5d6e57f05d6f806ULL,
      0xffffffffffffffffULL, 0xffffffffffffffffULL));
}

TEST_CASE("int256 arithmetic right shift negative", "[int256]") {
  REQUIRE((I256(-100) >> 1) == I256(-50));
}

TEST_CASE("int256 shift clamp", "[int256]") {
  REQUIRE((I256(1) << 256) == I256(0));
  REQUIRE((I256(-1) >> 256) == I256(-1));
  REQUIRE((I256(1) >> 256) == I256(0));
}

// ============================================================================
// Bitwise — boost-verified
// ============================================================================

TEST_CASE("int256 hardcoded bitwise", "[int256]") {
  auto full_pos = make256(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                          0x1111111111111111ull, 0x2222222222222222ull);
  auto full_neg = make256(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                          0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);

  REQUIRE((full_pos & full_neg) == make256(
      0x122416688abc9ab0ULL, 0x0000000000000000ULL,
      0x1010101010101010ULL, 0x0220022202202000ULL));

  REQUIRE((full_pos | full_neg) == make256(
      0xdebdfeffdafefefeULL, 0xffffffffffffffffULL,
      0xffddbb9977553311ULL, 0xabafeafe2baff22fULL));

  REQUIRE((full_pos ^ full_neg) == make256(
      0xcc99e8975042644eULL, 0xffffffffffffffffULL,
      0xefcdab8967452301ULL, 0xa98fe8dc298fd22fULL));

  REQUIRE(~full_pos == make256(
      0xedcba9876543210fULL, 0x0123456789abcdefULL,
      0xeeeeeeeeeeeeeeeeULL, 0xddddddddddddddddULL));

  REQUIRE(~full_neg == make256(
      0x2152411035014541ULL, 0xfedcba9876543210ULL,
      0x0123456789abcdefULL, 0x74523501f4520ff2ULL));
}

TEST_CASE("int256 bitwise identities", "[int256]") {
  auto a = make256(0x1234, 0x5678, 0x9ABC, 0xDEF0);
  REQUIRE(~~a == a);
  REQUIRE((a & ~a) == I256(0));
  REQUIRE((a | ~a) == I256(-1));
}

// ============================================================================
// Conversions
// ============================================================================

TEST_CASE("int256 explicit to int64", "[int256]") {
  REQUIRE(static_cast<int64_t>(I256(-42)) == -42);
}

TEST_CASE("int256 explicit to bool", "[int256]") {
  REQUIRE(static_cast<bool>(I256(1)));
  REQUIRE(!static_cast<bool>(I256(0)));
  REQUIRE(static_cast<bool>(I256(-1)));
}

TEST_CASE("int256 explicit to int128", "[int256]") {
  I128 v = (I128(1) << 100) + 42;
  REQUIRE(static_cast<I128>(I256(v)) == v);
}

TEST_CASE("int256 truncates to low bits", "[int256]") {
  auto big = make256(0xDEAD, 0xBEEF, 0xCAFE, 0xBABE);
  REQUIRE(static_cast<uint64_t>(big) == 0xDEAD);
}

// ============================================================================
// Compound assignment
// ============================================================================

TEST_CASE("int256 compound assignment", "[int256]") {
  I256 a(100);
  a += I256(50);
  REQUIRE(a == I256(150));
  a -= I256(200);
  REQUIRE(a == I256(-50));
  a *= I256(-2);
  REQUIRE(a == I256(100));
  a /= I256(3);
  REQUIRE(a == I256(33));
  a %= I256(10);
  REQUIRE(a == I256(3));
  a <<= 100;
  a >>= 100;
  REQUIRE(a == I256(3));
}

// ============================================================================
// Increment / decrement
// ============================================================================

TEST_CASE("int256 increment decrement", "[int256]") {
  I256 a(0);
  REQUIRE(++a == I256(1));
  REQUIRE(a++ == I256(1));
  REQUIRE(a == I256(2));
  REQUIRE(--a == I256(1));
  REQUIRE(a-- == I256(1));
  REQUIRE(a == I256(0));
  --a;
  REQUIRE(a == I256(-1));
  REQUIRE(a.is_negative());
}

// ============================================================================
// Algebraic identities with large values
// ============================================================================

TEST_CASE("int256 algebraic identities", "[int256]") {
  I256 vals[] = {
      make256(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
              0x1111111111111111ull, 0x2222222222222222ull),
      make256(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
              0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull),
      I256(1),
      I256(-1),
  };

  for (auto &a : vals) {
    REQUIRE((a + (-a)) == I256(0));
    REQUIRE(a * I256(1) == a);
    REQUIRE(a * I256(-1) == -a);
    REQUIRE(a * I256(0) == I256(0));
    for (auto &b : vals) {
      REQUIRE(a + b == b + a);
      REQUIRE(a * b == b * a);
      REQUIRE((a + b) - b == a);
    }
  }
}

TEST_CASE("int256 distributivity", "[int256]") {
  auto a = make256(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                   0x1111111111111111ull, 0x2222222222222222ull);
  auto b = make256(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                   0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);
  auto c = I256(1);
  REQUIRE(a * (b + c) == a * b + a * c);
}

// ============================================================================
// Predicate sign agreement
// ============================================================================

TEST_CASE("int256 predicate sign agreement with int128", "[int256]") {
  int64_t ax = 1000000000ll, ay = -2000000000ll, az = 3000000000ll;
  int64_t bx = -500000000ll, by = 1500000000ll, bz = -2500000000ll;
  int64_t cx = 750000000ll, cy = -1250000000ll, cz = 2250000000ll;

  I128 cross_x = I128(by) * cz - I128(bz) * cy;
  I128 cross_y = I128(bz) * cx - I128(bx) * cz;
  I128 cross_z = I128(bx) * cy - I128(by) * cx;

  I128 det128 = I128(ax) * cross_x + I128(ay) * cross_y + I128(az) * cross_z;
  I256 det256 = I256(ax) * I256(cross_x) + I256(ay) * I256(cross_y) +
                I256(az) * I256(cross_z);

  REQUIRE(to_i128(det256) == det128);

  int sign256 = det256.is_zero() ? 0 : (det256.is_negative() ? -1 : 1);
  int sign128 = (det128 == 0) ? 0 : (det128 < 0 ? -1 : 1);
  REQUIRE(sign256 == sign128);
}

TEST_CASE("int256 divmod: reconstruction oracle across widths and signs",
          "[int256]") {
  using tf::exact::int256;
  std::mt19937_64 rng(20260803);
  auto random_of_bits = [&](unsigned bits) -> int256 {
    int256 v(0);
    for (unsigned produced = 0; produced < bits; produced += 32) {
      const unsigned c = bits - produced < 32 ? bits - produced : 32;
      const auto chunk =
          rng() & ((std::uint64_t(1) << c) - (c == 64 ? 0 : 1));
      v = (v << c) + int256(chunk);
    }
    return v;
  };
  for (int it = 0; it < 20000; ++it) {
    const unsigned nb = 1 + unsigned(rng() % 255);
    const unsigned db = 1 + unsigned(rng() % nb);
    auto num = random_of_bits(nb);
    auto den = random_of_bits(db);
    if (den == int256(0))
      den = int256(1);
    if (rng() & 1)
      num = -num;
    if (rng() & 1)
      den = -den;
    const auto [q, r] = divmod(num, den);
    REQUIRE(q * den + r == num);
    // C++ semantics: remainder carries the dividend's sign, |r| < |den|
    const auto abs_r = r < int256(0) ? -r : r;
    const auto abs_den = den < int256(0) ? -den : den;
    REQUIRE(abs_r < abs_den);
    if (!(r == int256(0)))
      REQUIRE((r < int256(0)) == (num < int256(0)));
  }
  // the geological shape: degree-3 fraction magnitudes on the int64 ladder
  for (int it = 0; it < 2000; ++it) {
    const auto den = random_of_bits(160 + unsigned(rng() % 30));
    const auto num = random_of_bits(190 + unsigned(rng() % 30));
    if (den == int256(0))
      continue;
    const auto [q, r] = divmod(num, den);
    REQUIRE(q * den + r == num);
    REQUIRE(r < den);
    REQUIRE(!(r < int256(0)));
  }
}

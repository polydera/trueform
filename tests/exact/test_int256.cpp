/**
 * @file test_int256.cpp
 * @brief Tests for tf::exact::int256
 *
 * Covers construction, comparison, arithmetic, shifts, bitwise,
 * conversions, and large-value cross-checks. Results hardcoded from
 * boost::multiprecision verification.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/int128.hpp>
#include <trueform/exact/int256.hpp>

using I128 = tf::exact::int128;
using I256 = tf::exact::int256;

static auto to_i128(const I256 &v) -> I128 {
  return (I128(v.raw_limb(1)) << 64) | I128(v.raw_limb(0));
}

// ============================================================================
// Construction
// ============================================================================

TEST_CASE("int256 trivially default constructible", "[int256]") {
  STATIC_REQUIRE(std::is_trivially_default_constructible_v<I256>);
}

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
  REQUIRE(n.raw_limb(0) == ~uint64_t(0));
  REQUIRE(n.raw_limb(1) == ~uint64_t(0));
  REQUIRE(n.raw_limb(2) == ~uint64_t(0));
  REQUIRE(n.raw_limb(3) == ~uint64_t(0));
}

TEST_CASE("int256 from INT64_MIN", "[int256]") {
  I256 mn(INT64_MIN);
  REQUIRE(mn.is_negative());
  REQUIRE(static_cast<int64_t>(mn) == INT64_MIN);
  REQUIRE(mn.raw_limb(1) == ~uint64_t(0));
}

TEST_CASE("int256 from UINT64_MAX", "[int256]") {
  I256 u(uint64_t(UINT64_MAX));
  REQUIRE(!u.is_negative());
  REQUIRE(u.raw_limb(0) == UINT64_MAX);
  REQUIRE(u.raw_limb(1) == 0);
}

TEST_CASE("int256 from positive int128", "[int256]") {
  I128 v = I128(1) << 100;
  I256 w(v);
  REQUIRE(!w.is_negative());
  REQUIRE(to_i128(w) == v);
  REQUIRE(w.raw_limb(2) == 0);
  REQUIRE(w.raw_limb(3) == 0);
}

TEST_CASE("int256 from negative int128", "[int256]") {
  I128 v = -(I128(1) << 100);
  I256 w(v);
  REQUIRE(w.is_negative());
  REQUIRE(to_i128(w) == v);
  REQUIRE(w.raw_limb(2) == ~uint64_t(0));
  REQUIRE(w.raw_limb(3) == ~uint64_t(0));
}

TEST_CASE("int256 from 4 limbs", "[int256]") {
  I256 raw(1, 2, 3, 4);
  REQUIRE(raw.raw_limb(0) == 1);
  REQUIRE(raw.raw_limb(1) == 2);
  REQUIRE(raw.raw_limb(2) == 3);
  REQUIRE(raw.raw_limb(3) == 4);
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
// Addition & subtraction — verified against int128 + hardcoded from boost
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
  REQUIRE(sum.raw_limb(0) == 0);
  REQUIRE(sum.raw_limb(1) == (uint64_t(1) << 63));
  REQUIRE(sum.raw_limb(2) == 0);
  REQUIRE(sum.raw_limb(3) == 0);
  REQUIRE(sum / I256(2) == I256(big));
}

TEST_CASE("int256 hardcoded add", "[int256]") {
  I256 full_pos(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                0x1111111111111111ull, 0x2222222222222222ull);
  I256 full_neg(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);
  I256 result = full_pos + full_neg;
  REQUIRE(result ==
          I256(0xf0e2156865bb99aeULL, 0xffffffffffffffffULL,
               0x0fedcba987654321ULL, 0xadcfed202dd01230ULL));
}

TEST_CASE("int256 hardcoded sub", "[int256]") {
  I256 full_neg(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);
  I256 result = I256(0) - full_neg;
  REQUIRE(result ==
          I256(0x2152411035014542ULL, 0xfedcba9876543210ULL,
               0x0123456789abcdefULL, 0x74523501f4520ff2ULL));
}

TEST_CASE("int256 i64_min - i64_max", "[int256]") {
  I256 result = I256(INT64_MIN) - I256(INT64_MAX);
  REQUIRE(result ==
          I256(0x0000000000000001ULL, 0xffffffffffffffffULL,
               0xffffffffffffffffULL, 0xffffffffffffffffULL));
}

// ============================================================================
// Negation — hardcoded from boost
// ============================================================================

TEST_CASE("int256 hardcoded negate", "[int256]") {
  I256 full_pos(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                0x1111111111111111ull, 0x2222222222222222ull);
  REQUIRE(-full_pos ==
          I256(0xedcba98765432110ULL, 0x0123456789abcdefULL,
               0xeeeeeeeeeeeeeeeeULL, 0xddddddddddddddddULL));

  REQUIRE(-I256(1) ==
          I256(0xffffffffffffffffULL, 0xffffffffffffffffULL,
               0xffffffffffffffffULL, 0xffffffffffffffffULL));
}

// ============================================================================
// Multiplication — hardcoded from boost
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

TEST_CASE("int256 hardcoded mul full_pos * full_neg", "[int256]") {
  I256 full_pos(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                0x1111111111111111ull, 0x2222222222222222ull);
  I256 full_neg(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);
  REQUIRE(full_pos * full_neg ==
          I256(0xeb689f4ea447d620ULL, 0x7ccf959345111bc7ULL,
               0x7af77f6344f4e5fbULL, 0xd36dbd20f751d578ULL));
}

TEST_CASE("int256 hardcoded mul i128_big * i128_neg", "[int256]") {
  I256 a(I128(1) << 100);
  I256 b(-(I128(1) << 100));
  REQUIRE(a * b ==
          I256(0x0000000000000000ULL, 0x0000000000000000ULL,
               0x0000000000000000ULL, 0xffffffffffffff00ULL));
}

TEST_CASE("int256 hardcoded mul i64_max * i64_min", "[int256]") {
  REQUIRE(I256(INT64_MAX) * I256(INT64_MIN) ==
          I256(0x8000000000000000ULL, 0xc000000000000000ULL,
               0xffffffffffffffffULL, 0xffffffffffffffffULL));
}

TEST_CASE("int256 hardcoded mul u64_max * u64_max", "[int256]") {
  I256 u(uint64_t(UINT64_MAX));
  REQUIRE(u * u ==
          I256(0x0000000000000001ULL, 0xfffffffffffffffeULL,
               0x0000000000000000ULL, 0x0000000000000000ULL));
}

TEST_CASE("int256 widening mul 2^100 * 2^100 = 2^200", "[int256]") {
  I128 a = I128(1) << 100;
  I256 product = I256(a) * I256(a);
  REQUIRE(!product.is_negative());
  REQUIRE(product.raw_limb(3) == (uint64_t(1) << 8));
  REQUIRE(product.raw_limb(2) == 0);
  REQUIRE(product.raw_limb(1) == 0);
  REQUIRE(product.raw_limb(0) == 0);
}

// ============================================================================
// Division & modulo — hardcoded from boost
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

TEST_CASE("int256 hardcoded div full_pos / i64_max", "[int256]") {
  I256 full_pos(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                0x1111111111111111ull, 0x2222222222222222ull);
  REQUIRE(full_pos / I256(INT64_MAX) ==
          I256(0x530eca8641fdb97aULL, 0xaaaaaaaaaaaaaaadULL,
               0x4444444444444444ULL, 0x0000000000000000ULL));
  REQUIRE(full_pos % I256(INT64_MAX) ==
          I256(0x654320fedcba986aULL, 0x0000000000000000ULL,
               0x0000000000000000ULL, 0x0000000000000000ULL));
}

TEST_CASE("int256 hardcoded div full_neg / i128_big", "[int256]") {
  I256 full_neg(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);
  I256 i128_big(I128(1) << 100);
  REQUIRE(full_neg / i128_big ==
          I256(0x8765432100123457ULL, 0xe0badf00dfedcba9ULL,
               0xfffffffff8badcafULL, 0xffffffffffffffffULL));
  REQUIRE(full_neg % i128_big ==
          I256(0xdeadbeefcafebabeULL, 0xfffffff789abcdefULL,
               0xffffffffffffffffULL, 0xffffffffffffffffULL));
}

TEST_CASE("int256 hardcoded div full_pos / full_neg", "[int256]") {
  I256 full_pos(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                0x1111111111111111ull, 0x2222222222222222ull);
  I256 full_neg(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);
  REQUIRE(full_pos / full_neg == I256(0));
  REQUIRE(full_pos % full_neg == full_pos);
}

TEST_CASE("int256 large division round-trip", "[int256]") {
  I128 big = I128(1) << 100;
  I256 big200 = I256(big) * I256(big);
  REQUIRE(big200 / I256(big) == I256(big));
}

// ============================================================================
// Shifts — hardcoded from boost
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

TEST_CASE("int256 hardcoded shift left", "[int256]") {
  I256 full_pos(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                0x1111111111111111ull, 0x2222222222222222ull);
  REQUIRE((full_pos << 1) ==
          I256(0x2468acf13579bde0ULL, 0xfdb97530eca86420ULL,
               0x2222222222222223ULL, 0x4444444444444444ULL));

  REQUIRE((I256(1) << 200) ==
          I256(0x0000000000000000ULL, 0x0000000000000000ULL,
               0x0000000000000000ULL, 0x0000000000000100ULL));

  I256 i128_big(I128(1) << 100);
  REQUIRE((i128_big << 50) ==
          I256(0x0000000000000000ULL, 0x0000000000000000ULL,
               0x0000000000400000ULL, 0x0000000000000000ULL));
}

TEST_CASE("int256 hardcoded shift right", "[int256]") {
  I256 full_pos(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                0x1111111111111111ull, 0x2222222222222222ull);
  REQUIRE((full_pos >> 1) ==
          I256(0x091a2b3c4d5e6f78ULL, 0xff6e5d4c3b2a1908ULL,
               0x0888888888888888ULL, 0x1111111111111111ULL));

  I256 full_neg(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);
  REQUIRE((full_neg >> 3) ==
          I256(0xfbd5b7ddf95fd757ULL, 0x002468acf13579bdULL,
               0xbfdb97530eca8642ULL, 0xf175b95fc175be01ULL));
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
// Bitwise — hardcoded from boost
// ============================================================================

TEST_CASE("int256 hardcoded bitwise", "[int256]") {
  I256 full_pos(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
                0x1111111111111111ull, 0x2222222222222222ull);
  I256 full_neg(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
                0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);

  REQUIRE((full_pos & full_neg) ==
          I256(0x122416688abc9ab0ULL, 0x0000000000000000ULL,
               0x1010101010101010ULL, 0x0220022202202000ULL));
  REQUIRE((full_pos | full_neg) ==
          I256(0xdebdfeffdafefefeULL, 0xffffffffffffffffULL,
               0xffddbb9977553311ULL, 0xabafeafe2baff22fULL));
  REQUIRE((full_pos ^ full_neg) ==
          I256(0xcc99e8975042644eULL, 0xffffffffffffffffULL,
               0xefcdab8967452301ULL, 0xa98fe8dc298fd22fULL));
  REQUIRE(~full_pos ==
          I256(0xedcba9876543210fULL, 0x0123456789abcdefULL,
               0xeeeeeeeeeeeeeeeeULL, 0xddddddddddddddddULL));
  REQUIRE(~full_neg ==
          I256(0x2152411035014541ULL, 0xfedcba9876543210ULL,
               0x0123456789abcdefULL, 0x74523501f4520ff2ULL));
}

TEST_CASE("int256 bitwise identities", "[int256]") {
  I256 a(0x1234, 0x5678, 0x9ABC, 0xDEF0);
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

TEST_CASE("int256 truncates to low limb", "[int256]") {
  I256 big(0xDEAD, 0xBEEF, 0xCAFE, 0xBABE);
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
// Large-value cross-check via independent int128 reference
// ============================================================================

TEST_CASE("int256 independent reference mul", "[int256]") {
  // Independent schoolbook 4x4 multiply using tf::exact::uint128
  auto mul_ref = [](const I256 &a, const I256 &b) -> I256 {
    using u128 = tf::exact::uint128;
    uint64_t r[4] = {0, 0, 0, 0};
    for (unsigned i = 0; i < 4; ++i) {
      uint64_t carry = 0;
      for (unsigned j = 0; i + j < 4; ++j) {
        u128 prod = static_cast<u128>(a.raw_limb(i)) *
                    static_cast<u128>(b.raw_limb(j));
        u128 sum = static_cast<u128>(r[i + j]) + prod + carry;
        r[i + j] = static_cast<uint64_t>(sum);
        carry = static_cast<uint64_t>(sum >> 64);
      }
    }
    return I256(r[0], r[1], r[2], r[3]);
  };

  I256 vals[] = {
      I256(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
           0x1111111111111111ull, 0x2222222222222222ull),
      I256(0xAAAAAAAAAAAAAAAAull, 0x5555555555555555ull,
           0xCCCCCCCCCCCCCCCCull, 0x3333333333333333ull),
      I256(0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull,
           0xFFFFFFFFFFFFFFFFull, 0x7FFFFFFFFFFFFFFFull),
      I256(0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull,
           0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull),
      I256(1),
      I256(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
           0xFEDCBA9876543210ull, 0x0BADCAFE0BADF00Dull),
  };

  for (auto &a : vals)
    for (auto &b : vals)
      REQUIRE(a * b == mul_ref(a, b));
}

TEST_CASE("int256 algebraic identities", "[int256]") {
  I256 vals[] = {
      I256(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
           0x1111111111111111ull, 0x2222222222222222ull),
      I256(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
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
  I256 a(0x123456789ABCDEF0ull, 0xFEDCBA9876543210ull,
         0x1111111111111111ull, 0x2222222222222222ull);
  I256 b(0xDEADBEEFCAFEBABEull, 0x0123456789ABCDEFull,
         0xFEDCBA9876543210ull, 0x8BADCAFE0BADF00Dull);
  I256 c(1);
  REQUIRE(a * (b + c) == a * b + a * c);
}

// ============================================================================
// Orient3d-style predicate simulation — hardcoded from boost
// ============================================================================

TEST_CASE("int256 orient3d widening multiply", "[int256]") {
  I128 orient_a = (I128(1) << 100) + 12345;
  I128 orient_b = -(I128(1) << 95) + 67890;
  I256 product = I256(orient_a) * I256(orient_b);
  REQUIRE(product ==
          I256(0x0000000031f46c22ULL, 0x00107b0380000000ULL,
               0x0000000000000000ULL, 0xfffffffffffffff8ULL));
}

TEST_CASE("int256 widening div round-trip", "[int256]") {
  I128 orient_a = (I128(1) << 100) + 12345;
  I128 orient_b = -(I128(1) << 95) + 67890;
  I256 num = I256(orient_a) * I256(orient_b);
  I256 den = I256(orient_a);
  REQUIRE(num / den ==
          I256(0x0000000000010932ULL, 0xffffffff80000000ULL,
               0xffffffffffffffffULL, 0xffffffffffffffffULL));
  REQUIRE(num / den == I256(orient_b));
}

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

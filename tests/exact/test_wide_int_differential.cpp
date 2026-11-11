/**
 * @file test_wide_int_differential.cpp
 * @brief The wide integers against a truth on every platform.
 *
 * On MSVC every exact predicate runs on the portable 128-bit classes, and
 * int256 / int512 stand on whichever 128-bit type is selected. This suite
 * compiles with TF_FORCE_INT128_FALLBACK, so the classes are what it tests
 * everywhere: against a deliberately plain reference over 32-bit words —
 * schoolbook products, one-bit-at-a-time division — that needs no wide
 * type of its own, and additionally against the compiler's __int128 where
 * one exists. Inputs are mt19937_64 words taken directly, masked to a drawn
 * bit length, so small, medium and full-width magnitudes all occur.
 *
 * The conversions to double and to float are held to one reference built
 * the other way round: the magnitude's top significand bits carried by
 * `ldexp`, with the bits below them rounded half to even by hand. Every
 * width the type holds is swept, both signs, with the leading bit alone,
 * all ones, and the ties either side of the last bit each significand
 * keeps — a float tie reached through a double is a different value, so
 * one rounding is the whole claim.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>

#if !defined(_MSC_VER) && !defined(TF_FORCE_INT128_FALLBACK)
#error                                                                         \
    "the suite tests the portable classes: build it with TF_FORCE_INT128_FALLBACK"
#endif

#include <trueform/exact/int128.hpp>
#include <trueform/exact/int256.hpp>
#include <trueform/exact/int512.hpp>
#include <trueform/exact/wide_from_double.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

using differential_u128 = tf::exact::uint128;
using differential_i128 = tf::exact::int128;
using differential_i256 = tf::exact::int256;
using differential_i512 = tf::exact::int512;

/// A two's-complement value of `L` 32-bit words, least significant first.
template <int L> struct differential_ref {
  std::uint32_t w[L] = {};
  static constexpr unsigned bits = 32u * L;

  auto operator==(const differential_ref &o) const -> bool {
    for (int i = 0; i < L; ++i)
      if (w[i] != o.w[i])
        return false;
    return true;
  }
  auto operator!=(const differential_ref &o) const -> bool {
    return !(*this == o);
  }
};

template <int L>
auto differential_negative(const differential_ref<L> &a) -> bool {
  return (a.w[L - 1] >> 31) != 0;
}

template <int L>
auto differential_of_u64(std::uint64_t v) -> differential_ref<L> {
  differential_ref<L> r;
  r.w[0] = std::uint32_t(v);
  r.w[1] = std::uint32_t(v >> 32);
  return r;
}

template <int L>
auto differential_add(const differential_ref<L> &a,
                      const differential_ref<L> &b) -> differential_ref<L> {
  differential_ref<L> r;
  std::uint64_t carry = 0;
  for (int i = 0; i < L; ++i) {
    carry += std::uint64_t(a.w[i]) + b.w[i];
    r.w[i] = std::uint32_t(carry);
    carry >>= 32;
  }
  return r;
}

template <int L>
auto differential_not(const differential_ref<L> &a) -> differential_ref<L> {
  differential_ref<L> r;
  for (int i = 0; i < L; ++i)
    r.w[i] = ~a.w[i];
  return r;
}

template <int L>
auto differential_neg(const differential_ref<L> &a) -> differential_ref<L> {
  return differential_add(differential_not(a), differential_of_u64<L>(1));
}

template <int L>
auto differential_sub(const differential_ref<L> &a,
                      const differential_ref<L> &b) -> differential_ref<L> {
  return differential_add(a, differential_neg(b));
}

template <int L>
auto differential_mul(const differential_ref<L> &a,
                      const differential_ref<L> &b) -> differential_ref<L> {
  differential_ref<L> r;
  for (int i = 0; i < L; ++i) {
    std::uint64_t carry = 0;
    for (int j = 0; i + j < L; ++j) {
      carry += std::uint64_t(a.w[i]) * b.w[j] + r.w[i + j];
      r.w[i + j] = std::uint32_t(carry);
      carry >>= 32;
    }
  }
  return r;
}

template <int L>
auto differential_bit(const differential_ref<L> &a, unsigned k) -> bool {
  return ((a.w[k / 32] >> (k % 32)) & 1u) != 0;
}

template <int L>
auto differential_shl(const differential_ref<L> &a, unsigned s)
    -> differential_ref<L> {
  differential_ref<L> r;
  for (unsigned k = 0; k + s < differential_ref<L>::bits; ++k)
    if (differential_bit(a, k))
      r.w[(k + s) / 32] |= std::uint32_t(1) << ((k + s) % 32);
  return r;
}

template <int L>
auto differential_shr(const differential_ref<L> &a, unsigned s, bool arithmetic)
    -> differential_ref<L> {
  differential_ref<L> r;
  const bool fill = arithmetic && differential_negative(a);
  for (unsigned k = 0; k < differential_ref<L>::bits; ++k) {
    const unsigned source = k + s;
    const bool v =
        source < differential_ref<L>::bits ? differential_bit(a, source) : fill;
    if (v)
      r.w[k / 32] |= std::uint32_t(1) << (k % 32);
  }
  return r;
}

template <int L>
auto differential_ucmp(const differential_ref<L> &a,
                       const differential_ref<L> &b) -> int {
  for (int i = L; i-- > 0;)
    if (a.w[i] != b.w[i])
      return a.w[i] < b.w[i] ? -1 : 1;
  return 0;
}

template <int L>
auto differential_less(const differential_ref<L> &a,
                       const differential_ref<L> &b) -> bool {
  if (differential_negative(a) != differential_negative(b))
    return differential_negative(a);
  return differential_ucmp(a, b) < 0;
}

/// Truncating signed division, one quotient bit at a time.
template <int L>
auto differential_divmod(const differential_ref<L> &a,
                         const differential_ref<L> &b, differential_ref<L> &q,
                         differential_ref<L> &r) -> void {
  const auto n = differential_negative(a) ? differential_neg(a) : a;
  const auto d = differential_negative(b) ? differential_neg(b) : b;
  q = {};
  r = {};
  for (unsigned k = differential_ref<L>::bits; k-- > 0;) {
    for (int i = L; i-- > 1;)
      r.w[i] = (r.w[i] << 1) | (r.w[i - 1] >> 31);
    r.w[0] = (r.w[0] << 1) | (differential_bit(n, k) ? 1u : 0u);
    if (differential_ucmp(r, d) >= 0) {
      r = differential_sub(r, d);
      q.w[k / 32] |= std::uint32_t(1) << (k % 32);
    }
  }
  if (differential_negative(a) != differential_negative(b))
    q = differential_neg(q);
  if (differential_negative(a))
    r = differential_neg(r);
}

/// The correctly rounded `Real` of a magnitude, built bit by bit: one of at
/// most `Real`'s significand is the sum of its own powers, and a wider one
/// is its top significand bits carried by `ldexp`, the bit below them
/// deciding with what lies under it — above a half ulp rounds away, exactly
/// a half rounds to even. A magnitude past the type's range scales to
/// infinity. It shares no step with the library's conversion.
template <typename Real, int L>
auto differential_magnitude_real(const differential_ref<L> &m) -> Real {
  constexpr unsigned significand = unsigned(std::numeric_limits<Real>::digits);
  unsigned width = 0;
  for (unsigned k = 0; k < differential_ref<L>::bits; ++k)
    if (differential_bit(m, k))
      width = k + 1;
  if (width <= significand) {
    Real magnitude = Real(0);
    for (unsigned k = 0; k < width; ++k)
      if (differential_bit(m, k))
        magnitude += std::ldexp(Real(1), int(k));
    return magnitude;
  }
  const unsigned dropped = width - significand;
  std::uint64_t kept = 0;
  for (unsigned k = 0; k < significand; ++k)
    if (differential_bit(m, dropped + k))
      kept |= std::uint64_t(1) << k;
  bool under = false;
  for (unsigned k = 0; k + 1 < dropped; ++k)
    under = under || differential_bit(m, k);
  if (differential_bit(m, dropped - 1))
    kept += under ? 1 : (kept & 1);
  return std::ldexp(Real(kept), int(dropped));
}

/// The value's own magnitude converted, signed.
template <typename Real, int L>
auto differential_real(const differential_ref<L> &a) -> Real {
  const bool negative = differential_negative(a);
  const Real at =
      differential_magnitude_real<Real>(negative ? differential_neg(a) : a);
  return negative ? -at : at;
}

template <int L>
auto differential_from_double(double d) -> differential_ref<L> {
  const double t = std::trunc(d);
  int exponent = 0;
  const double fraction = std::frexp(std::fabs(t), &exponent);
  if (t == 0.0)
    return {};
  const auto mantissa =
      differential_of_u64<L>(std::uint64_t(std::ldexp(fraction, 53)));
  const auto m =
      exponent >= 53
          ? differential_shl(mantissa, unsigned(exponent - 53))
          : differential_shr(mantissa, unsigned(53 - exponent), false);
  return t < 0.0 ? differential_neg(m) : m;
}

auto differential_words_of(const differential_u128 &v, std::uint32_t *out)
    -> void {
  const auto lo = static_cast<std::uint64_t>(v);
  const auto hi = static_cast<std::uint64_t>(v >> 64);
  out[0] = std::uint32_t(lo);
  out[1] = std::uint32_t(lo >> 32);
  out[2] = std::uint32_t(hi);
  out[3] = std::uint32_t(hi >> 32);
}

auto differential_limb(const std::uint32_t *in) -> differential_u128 {
  const std::uint64_t lo = std::uint64_t(in[0]) | (std::uint64_t(in[1]) << 32);
  const std::uint64_t hi = std::uint64_t(in[2]) | (std::uint64_t(in[3]) << 32);
  return (differential_u128(hi) << 64) | differential_u128(lo);
}

/// The class value and its reference, the same bits in both.
template <typename W> struct differential_glue;

template <> struct differential_glue<differential_i128> {
  static constexpr int L = 4;
  static auto of(const differential_ref<4> &r) -> differential_i128 {
    return differential_i128(differential_limb(r.w));
  }
  static auto ref_of(const differential_i128 &v) -> differential_ref<4> {
    differential_ref<4> r;
    differential_words_of(differential_u128(v), r.w);
    return r;
  }
};

template <> struct differential_glue<differential_i256> {
  static constexpr int L = 8;
  static auto of(const differential_ref<8> &r) -> differential_i256 {
    return differential_i256(differential_limb(r.w),
                             differential_limb(r.w + 4));
  }
  static auto ref_of(const differential_i256 &v) -> differential_ref<8> {
    differential_ref<8> r;
    differential_words_of(v.lo(), r.w);
    differential_words_of(v.hi(), r.w + 4);
    return r;
  }
};

template <> struct differential_glue<differential_i512> {
  static constexpr int L = 16;
  static auto of(const differential_ref<16> &r) -> differential_i512 {
    return differential_i512(differential_limb(r.w), differential_limb(r.w + 4),
                             differential_limb(r.w + 8),
                             differential_limb(r.w + 12));
  }
  static auto ref_of(const differential_i512 &v) -> differential_ref<16> {
    differential_ref<16> r;
    for (std::size_t k = 0; k < 4; ++k)
      differential_words_of(v.limb(k), r.w + 4 * k);
    return r;
  }
};

/// Words of the drawn bit length, negated on a drawn bit.
template <int L> struct differential_stream {
  std::mt19937_64 rng;

  explicit differential_stream(std::uint64_t seed) : rng(seed) {}

  auto next() -> differential_ref<L> {
    const std::uint64_t control = rng();
    differential_ref<L> v;
    for (int i = 0; i < L; i += 2) {
      const std::uint64_t word = rng();
      v.w[i] = std::uint32_t(word);
      v.w[i + 1] = std::uint32_t(word >> 32);
    }
    const unsigned width = unsigned(control % (differential_ref<L>::bits + 1));
    v = differential_shr(differential_shl(v, differential_ref<L>::bits - width),
                         differential_ref<L>::bits - width, false);
    if (width == 0)
      v = {};
    return ((control >> 32) & 1u) != 0 ? differential_neg(v) : v;
  }
};

/// Zero, one, two, every digit boundary the classes split on one either
/// side — 32-bit words up to int256, 64-bit words for int512 — and the
/// extremes, each with its negation.
template <int L> auto differential_edges() -> std::vector<differential_ref<L>> {
  std::vector<differential_ref<L>> out;
  const auto both = [&](const differential_ref<L> &v) {
    out.push_back(v);
    out.push_back(differential_neg(v));
  };
  both({});
  both(differential_of_u64<L>(1));
  both(differential_of_u64<L>(2));
  const unsigned step = L <= 8 ? 32u : 64u;
  for (unsigned k = 0; k < differential_ref<L>::bits; k += step)
    for (const unsigned at : {k, k + step - 1}) {
      const auto p = differential_shl(differential_of_u64<L>(1), at);
      both(p);
      both(differential_add(p, differential_of_u64<L>(1)));
      both(differential_sub(p, differential_of_u64<L>(1)));
    }
  auto max = differential_not(differential_ref<L>{});
  max.w[L - 1] >>= 1;
  both(max);
  both(differential_add(max, differential_of_u64<L>(1)));
  return out;
}

template <typename W>
auto differential_check_pair(const differential_ref<differential_glue<W>::L> &a,
                             const differential_ref<differential_glue<W>::L> &b)
    -> void {
  using G = differential_glue<W>;
  constexpr int L = G::L;
  const W x = G::of(a), y = G::of(b);
  REQUIRE(G::ref_of(x + y) == differential_add(a, b));
  REQUIRE(G::ref_of(x - y) == differential_sub(a, b));
  REQUIRE(G::ref_of(x * y) == differential_mul(a, b));
  REQUIRE(G::ref_of(-x) == differential_neg(a));
  REQUIRE(G::ref_of(~x) == differential_not(a));
  REQUIRE((x < y) == differential_less(a, b));
  REQUIRE((x > y) == differential_less(b, a));
  REQUIRE((x <= y) == !differential_less(b, a));
  REQUIRE((x >= y) == !differential_less(a, b));
  REQUIRE((x == y) == (a == b));
  REQUIRE((x != y) == (a != b));
  differential_ref<L> both, either, one;
  for (int i = 0; i < L; ++i) {
    both.w[i] = a.w[i] & b.w[i];
    either.w[i] = a.w[i] | b.w[i];
    one.w[i] = a.w[i] ^ b.w[i];
  }
  REQUIRE(G::ref_of(x & y) == both);
  REQUIRE(G::ref_of(x | y) == either);
  REQUIRE(G::ref_of(x ^ y) == one);
  W t = x;
  t += y;
  REQUIRE(G::ref_of(t) == differential_add(a, b));
  t = x;
  t *= y;
  REQUIRE(G::ref_of(t) == differential_mul(a, b));
  const unsigned s = b.w[0] % differential_ref<L>::bits;
  REQUIRE(G::ref_of(x << s) == differential_shl(a, s));
  REQUIRE(G::ref_of(x >> s) == differential_shr(a, s, true));
  if (b != differential_ref<L>{}) {
    differential_ref<L> q, r;
    differential_divmod(a, b, q, r);
    REQUIRE(G::ref_of(x / y) == q);
    REQUIRE(G::ref_of(x % y) == r);
  }
}

template <typename W>
auto differential_check_value(
    const differential_ref<differential_glue<W>::L> &a) -> void {
  using G = differential_glue<W>;
  const W x = G::of(a);
  const double got = double(x), want = differential_real<double>(a);
  REQUIRE(std::memcmp(&got, &want, sizeof(double)) == 0);
  if constexpr (differential_glue<W>::L == 4) {
    const differential_u128 magnitude = differential_limb(a.w);
    const float signed_got = float(x),
                signed_want = differential_real<float>(a);
    REQUIRE(std::memcmp(&signed_got, &signed_want, sizeof(float)) == 0);
    const float plain_got = float(magnitude),
                plain_want = differential_magnitude_real<float>(a);
    REQUIRE(std::memcmp(&plain_got, &plain_want, sizeof(float)) == 0);
    const double wide_got = double(magnitude),
                 wide_want = differential_magnitude_real<double>(a);
    REQUIRE(std::memcmp(&wide_got, &wide_want, sizeof(double)) == 0);
  }
  REQUIRE(bool(x) == (a != differential_ref<G::L>{}));
  REQUIRE(static_cast<std::uint64_t>(x) ==
          (std::uint64_t(a.w[0]) | (std::uint64_t(a.w[1]) << 32)));
}

/// Every significant width the type holds, and per width: the leading bit
/// alone, all ones (the carry into the next binade), and per significand the
/// type converts into the tie at half an ulp below an even and an odd last
/// bit and the tie with one more bit far below — each with its negation, and
/// the extremes after them.
template <typename W> auto differential_conversion_edges() -> void {
  constexpr int L = differential_glue<W>::L;
  const auto one = differential_of_u64<L>(1);
  for (unsigned width = 1; width <= differential_ref<L>::bits; ++width) {
    const auto lead = differential_shl(one, width - 1);
    std::vector<differential_ref<L>> forms{
        lead, differential_sub(differential_shl(one, width), one)};
    for (const unsigned significand : {24u, 53u}) {
      if (width <= significand + 1)
        continue;
      const auto half = differential_shl(one, width - significand - 1);
      const auto ulp = differential_shl(one, width - significand);
      forms.push_back(differential_add(lead, half));
      forms.push_back(differential_add(differential_add(lead, ulp), half));
      forms.push_back(differential_add(differential_add(lead, half), one));
      forms.push_back(differential_sub(differential_add(lead, half), one));
    }
    for (const auto &v : forms) {
      differential_check_value<W>(v);
      differential_check_value<W>(differential_neg(v));
    }
  }
  auto max = differential_not(differential_ref<L>{});
  max.w[L - 1] >>= 1;
  differential_check_value<W>(max);
  differential_check_value<W>(differential_neg(max));
  differential_check_value<W>(differential_add(max, one));
}

template <typename W>
auto differential_sweep(std::uint64_t seed, int pairs) -> void {
  differential_conversion_edges<W>();
  constexpr int L = differential_glue<W>::L;
  const auto edges = differential_edges<L>();
  for (const auto &a : edges)
    for (const auto &b : edges)
      differential_check_pair<W>(a, b);
  differential_stream<L> stream(seed);
  for (int i = 0; i < pairs; ++i) {
    const auto a = stream.next();
    differential_check_pair<W>(a, stream.next());
    if (i % 8 == 0)
      differential_check_value<W>(a);
  }
}

} // namespace

TEST_CASE("wide int: a 128-bit value converts to double rounded once",
          "[int128][differential]") {
  // 2^64 + 2^63 + 2^11 + 1: rounding the low limb first lands on the tie
  // 2^64 + 2^63 + 2^11, which rounds to even and loses the 1 that makes the
  // nearest double 2^64 + 2^63 + 2^12.
  const std::uint64_t lo = 0x8000000000000801ull;
  const double nearest =
      std::ldexp(1.0, 64) + std::ldexp(1.0, 63) + std::ldexp(1.0, 12);
  REQUIRE(double(tf::exact::uint128(lo, 1)) == nearest);
  REQUIRE(double(tf::exact::int128(lo, 1)) == nearest);
  REQUIRE(double(-tf::exact::int128(lo, 1)) == -nearest);
}

TEST_CASE("wide int: a 128-bit value converts to float rounded once",
          "[int128][differential]") {
  // 2^126 + 2^102 + 1: its double is the tie 2^126 + 2^102, which rounds to
  // even and loses the 1 that makes the nearest float 2^126 + 2^103, so a
  // conversion routed through a double states a different value.
  const auto value = (tf::exact::uint128(1) << 126) |
                     (tf::exact::uint128(1) << 102) | tf::exact::uint128(1);
  const float nearest = std::ldexp(1.0f, 126) + std::ldexp(1.0f, 103);
  REQUIRE(float(value) == nearest);
  REQUIRE(float(tf::exact::int128(value)) == nearest);
  REQUIRE(float(-tf::exact::int128(value)) == -nearest);
  REQUIRE(float(double(value)) != nearest);
  // The signed extremes stand inside the float's range; an unsigned
  // magnitude past it is infinite, which is what a native conversion of
  // such a value states.
  const auto highest = ~tf::exact::uint128(0);
  REQUIRE(float(tf::exact::int128(tf::exact::uint128(1) << 127)) ==
          -std::ldexp(1.0f, 127));
  REQUIRE(float(tf::exact::int128(highest >> 1)) == std::ldexp(1.0f, 127));
  REQUIRE(std::isinf(float(highest)));
  REQUIRE(float(highest) > 0.0f);
}

TEST_CASE("wide int: a 256- or 512-bit value converts to double rounded once",
          "[int256][int512][differential]") {
  // 2^128 + 2^127 + 2^75 + 1: rounding the low half first lands on the tie
  // 2^128 + 2^127 + 2^75, which rounds to even and loses the 1 that makes
  // the nearest double 2^128 + 2^127 + 2^76.
  const auto low = (tf::exact::uint128(1) << 127) |
                   (tf::exact::uint128(1) << 75) | tf::exact::uint128(1);
  const double nearest =
      std::ldexp(1.0, 128) + std::ldexp(1.0, 127) + std::ldexp(1.0, 76);
  const tf::exact::int256 a(low, tf::exact::uint128(1));
  const tf::exact::int512 b(low, tf::exact::uint128(1), tf::exact::uint128(0),
                            tf::exact::uint128(0));
  REQUIRE(double(a) == nearest);
  REQUIRE(double(-a) == -nearest);
  REQUIRE(double(b) == nearest);
  REQUIRE(double(-b) == -nearest);
}

TEST_CASE("wide int: an unsigned 128-bit value widens by zero extension",
          "[int256][int512][differential]") {
  const auto top = tf::exact::uint128(1) << 127;
  const auto ones = ~tf::exact::uint128(0);
  const tf::exact::int256 a(top);
  const tf::exact::int512 b(ones);
  REQUIRE(a.lo() == top);
  REQUIRE(a.hi() == tf::exact::uint128(0));
  REQUIRE(!a.is_negative());
  REQUIRE(b.limb(0) == ones);
  REQUIRE(b.limb(1) == tf::exact::uint128(0));
  REQUIRE(b.limb(3) == tf::exact::uint128(0));
  REQUIRE(!b.is_negative());
}

TEST_CASE("wide int: int128 against the word reference",
          "[int128][differential]") {
  differential_sweep<differential_i128>(0x128128128ull, 20000);
  differential_stream<4> stream(0xd0b1e128ull);
  for (int i = 0; i < 2000; ++i) {
    const auto want = stream.next();
    const double d = differential_real<double>(want);
    if (std::fabs(d) < std::ldexp(1.0, 127))
      REQUIRE(differential_glue<differential_i128>::ref_of(
                  tf::exact::wide_from_double<differential_i128>(d)) ==
              differential_from_double<4>(d));
  }
}

TEST_CASE("wide int: int256 against the word reference",
          "[int256][differential]") {
  differential_sweep<differential_i256>(0x256256256ull, 5000);
}

TEST_CASE("wide int: int512 against the word reference",
          "[int512][differential]") {
  differential_sweep<differential_i512>(0x512512512ull, 1500);
}

#if !defined(_MSC_VER) && (defined(__GNUC__) || defined(__clang__))
TEST_CASE("wide int: the 128-bit classes against the compiler's own type",
          "[int128][differential]") {
  using native_u = unsigned __int128;
  using native_i = __int128;
  const auto bits_of = [](native_u v) {
    return differential_u128(std::uint64_t(v), std::uint64_t(v >> 64));
  };
  const auto words_of = [](const differential_u128 &v) {
    return (native_u(v.hi()) << 64) | v.lo();
  };
  differential_stream<4> stream(0xc1a55128ull);
  for (int i = 0; i < 20000; ++i) {
    const auto ra = stream.next(), rb = stream.next();
    const native_u ua = words_of(differential_limb(ra.w));
    const native_u ub = words_of(differential_limb(rb.w));
    const native_i a = native_i(ua), b = native_i(ub);
    const differential_i128 x(bits_of(ua)), y(bits_of(ub));
    REQUIRE(words_of(differential_u128(x * y)) == ua * ub);
    REQUIRE(words_of(differential_u128(x + y)) == ua + ub);
    REQUIRE(words_of(differential_u128(x - y)) == ua - ub);
    REQUIRE((x < y) == (a < b));
    REQUIRE(double(x) == double(a));
    REQUIRE(double(bits_of(ua)) == double(ua));
    REQUIRE(float(x) == float(a));
    REQUIRE(float(bits_of(ua)) == float(ua));
    const unsigned s = unsigned(ub % 128);
    REQUIRE(words_of(differential_u128(x >> s)) == native_u(a >> s));
    if (b != 0 && !(b == -1 && a == native_i(native_u(1) << 127))) {
      REQUIRE(words_of(differential_u128(x / y)) == native_u(a / b));
      REQUIRE(words_of(differential_u128(x % y)) == native_u(a % b));
      REQUIRE(words_of(bits_of(ua) / bits_of(ub)) == ua / ub);
      REQUIRE(words_of(bits_of(ua) % bits_of(ub)) == ua % ub);
    }
  }
}
#endif

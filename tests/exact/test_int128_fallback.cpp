/**
 * @file test_int128_fallback.cpp
 * @brief The portable 128-bit classes against the compiler's own type.
 *
 * On MSVC there is no native __int128 and tf::exact provides hand-rolled
 * classes; everywhere else they are unreachable, so nothing exercises them
 * on the machines the library is developed on. TF_FORCE_INT128_FALLBACK
 * compiles them beside the native type, and this suite is what the macro
 * exists for: every quotient and remainder the fallback states, against the
 * compiler's own, over a sweep of the widths its two division paths split
 * on — a divisor inside 32 bits takes the schoolbook digits, anything wider
 * takes the estimate-and-correct loop.
 *
 * The macro is a whole-translation-unit switch, so this source is compiled
 * alone: with it, tf::exact::uint128 is the class, and in every other unit
 * of the same binary it is the compiler's type.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>

#if defined(TF_FORCE_INT128_FALLBACK) && !defined(_MSC_VER) &&                 \
    (defined(__GNUC__) || defined(__clang__))

#include <trueform/exact/int128.hpp>

#include <cstdint>
#include <random>

namespace {

using fallback_u = tf::exact::uint128;
using fallback_i = tf::exact::int128;
using native_u = unsigned __int128;
using native_i = __int128;

/// A 128-bit value as its two words — the one form both implementations
/// answer in, whatever they store.
struct fallback_words {
  std::uint64_t lo = 0;
  std::uint64_t hi = 0;
  auto operator==(const fallback_words &o) const -> bool {
    return lo == o.lo && hi == o.hi;
  }
};

auto fallback_words_of(const fallback_u &v) -> fallback_words {
  return {v.lo(), v.hi()};
}

auto fallback_words_of(const fallback_i &v) -> fallback_words {
  return {v.lo(), v.hi()};
}

auto fallback_words_of(native_u v) -> fallback_words {
  return {static_cast<std::uint64_t>(v), static_cast<std::uint64_t>(v >> 64)};
}

auto fallback_words_of(native_i v) -> fallback_words {
  return fallback_words_of(static_cast<native_u>(v));
}

/// The same bits in both implementations.
struct fallback_pair {
  fallback_u portable;
  native_u compiler;
};

auto fallback_pair_of(const fallback_words &w) -> fallback_pair {
  return {fallback_u(w.lo, w.hi),
          static_cast<native_u>(w.lo) |
              (static_cast<native_u>(w.hi) << 64)};
}

/// A deterministic value of at most `bits` significant bits.
struct fallback_stream {
  std::mt19937_64 rng{20260909};

  auto words_of_bits(unsigned bits) -> fallback_words {
    if (bits == 0)
      return {};
    const native_u mask = bits >= 128
                              ? ~static_cast<native_u>(0)
                              : ((static_cast<native_u>(1) << bits) -
                                 static_cast<native_u>(1));
    const auto v = ((static_cast<native_u>(rng()) << 64) |
                    static_cast<native_u>(rng())) &
                   mask;
    return fallback_words_of(v);
  }
};

auto check_unsigned_divmod(const fallback_words &n, const fallback_words &d)
    -> void {
  const auto num = fallback_pair_of(n);
  const auto den = fallback_pair_of(d);
  REQUIRE(fallback_words_of(num.portable / den.portable) ==
          fallback_words_of(num.compiler / den.compiler));
  REQUIRE(fallback_words_of(num.portable % den.portable) ==
          fallback_words_of(num.compiler % den.compiler));
}

auto check_signed_divmod(const fallback_words &n, const fallback_words &d)
    -> void {
  const fallback_i portable_num(n.lo, n.hi), portable_den(d.lo, d.hi);
  const auto compiler_num = static_cast<native_i>(
      static_cast<native_u>(n.lo) | (static_cast<native_u>(n.hi) << 64));
  const auto compiler_den = static_cast<native_i>(
      static_cast<native_u>(d.lo) | (static_cast<native_u>(d.hi) << 64));
  REQUIRE(fallback_words_of(portable_num / portable_den) ==
          fallback_words_of(compiler_num / compiler_den));
  REQUIRE(fallback_words_of(portable_num % portable_den) ==
          fallback_words_of(compiler_num % compiler_den));
}

} // namespace

TEST_CASE("int128 fallback: unsigned divmod over every width pair",
          "[int128][fallback]") {
  fallback_stream stream;
  for (unsigned nbits = 1; nbits <= 128; ++nbits)
    for (unsigned dbits = 1; dbits <= nbits; ++dbits)
      for (int sample = 0; sample < 2; ++sample) {
        const auto n = stream.words_of_bits(nbits);
        auto d = stream.words_of_bits(dbits);
        if (d.lo == 0 && d.hi == 0)
          d.lo = 1;
        check_unsigned_divmod(n, d);
      }
}

TEST_CASE("int128 fallback: signed divmod over every width pair and sign",
          "[int128][fallback]") {
  fallback_stream stream;
  for (unsigned nbits = 1; nbits <= 127; ++nbits)
    for (unsigned dbits = 1; dbits <= nbits; ++dbits) {
      const auto n = stream.words_of_bits(nbits);
      auto d = stream.words_of_bits(dbits);
      if (d.lo == 0 && d.hi == 0)
        d.lo = 1;
      const auto negated = [](const fallback_words &w) {
        return fallback_words_of(
            -(static_cast<native_i>(static_cast<native_u>(w.lo) |
                                    (static_cast<native_u>(w.hi) << 64))));
      };
      check_signed_divmod(n, d);
      check_signed_divmod(negated(n), d);
      check_signed_divmod(n, negated(d));
      check_signed_divmod(negated(n), negated(d));
    }
}

TEST_CASE("int128 fallback: divmod at the boundaries of both paths",
          "[int128][fallback]") {
  const fallback_words zero{0, 0};
  const fallback_words one{1, 0};
  const fallback_words all_ones{~0ull, ~0ull};
  const fallback_words top_bit{0, 0x8000000000000000ull};
  const fallback_words just_narrow{0xffffffffull, 0};      // 32 bits
  const fallback_words just_wide{0x100000000ull, 0};       // 33 bits
  const fallback_words word{~0ull, 0};                     // 64 bits
  const fallback_words word_and_one{0, 1};                 // 65 bits

  const fallback_words values[] = {zero,       one,  all_ones,  top_bit,
                                   just_narrow, just_wide, word, word_and_one};
  for (const auto &n : values)
    for (const auto &d : values) {
      if (d.lo == 0 && d.hi == 0)
        continue;
      check_unsigned_divmod(n, d);
    }

  // the signed extremes, where the magnitude of the most negative value
  // carries the sign bit
  check_signed_divmod(top_bit, one);
  check_signed_divmod(top_bit, {7, 0});
  check_signed_divmod(top_bit, word_and_one);
  check_signed_divmod(top_bit, top_bit);
  check_signed_divmod(all_ones, top_bit);
}

TEST_CASE("int128 fallback: the ring agrees with the compiler",
          "[int128][fallback]") {
  fallback_stream stream;
  for (int it = 0; it < 4000; ++it) {
    const auto a = fallback_pair_of(stream.words_of_bits(128));
    const auto b = fallback_pair_of(stream.words_of_bits(128));
    const unsigned s = unsigned(stream.rng() % 128);
    REQUIRE(fallback_words_of(a.portable + b.portable) ==
            fallback_words_of(native_u(a.compiler + b.compiler)));
    REQUIRE(fallback_words_of(a.portable - b.portable) ==
            fallback_words_of(native_u(a.compiler - b.compiler)));
    REQUIRE(fallback_words_of(a.portable * b.portable) ==
            fallback_words_of(native_u(a.compiler * b.compiler)));
    REQUIRE(fallback_words_of(a.portable << s) ==
            fallback_words_of(native_u(a.compiler << s)));
    REQUIRE(fallback_words_of(a.portable >> s) ==
            fallback_words_of(native_u(a.compiler >> s)));
    REQUIRE((a.portable < b.portable) == (a.compiler < b.compiler));
    REQUIRE((a.portable == b.portable) == (a.compiler == b.compiler));
  }
}

#endif

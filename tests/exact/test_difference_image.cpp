/**
 * @file test_difference_image.cpp
 * @brief tf::exact::difference_image — what a filter may assume of a
 *        coordinate difference it reads in a double.
 *
 * The three verdicts, each against a reference that shares no step with the
 * function: the overflow is decided from the type's own bounds, the kind
 * from the exact difference against 2^52, and the image is built the other
 * way round — the magnitude's top 53 bits carried by `ldexp`, with the bits
 * below them rounded half to even by hand. Both coordinates that reach the
 * pattern route run: the 64-bit integer, and the 128-bit type whichever one
 * this translation unit holds.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "uniform_draw.hpp"
#include <catch2/catch_test_macros.hpp>
#include <trueform/exact/bits/twos_complement.hpp>
#include <trueform/exact/difference_image.hpp>
#include <trueform/exact/int128.hpp>
#include <trueform/exact/int64.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

template <typename Coord>
using image_pattern = typename tf::exact::bits::bit_pattern<Coord>::type;

template <typename Coord> constexpr auto image_width() -> unsigned {
  return unsigned(sizeof(Coord)) * 8u;
}

/// The value of a two's-complement pattern, which the test states its
/// operands and their magnitudes in.
template <typename Coord>
auto image_valued(const image_pattern<Coord> &bits) -> Coord {
  return tf::exact::bits::signed_value<Coord>(bits);
}

template <typename Coord> auto image_bit(const Coord &v, unsigned k) -> bool {
  using U = image_pattern<Coord>;
  return U(U(U(v) >> k) & U(1)) != U(0);
}

/// The value's magnitude, read in its own pattern so the lowest value of
/// the type has one.
template <typename Coord>
auto image_magnitude(const Coord &v) -> image_pattern<Coord> {
  using U = image_pattern<Coord>;
  return v < Coord(0) ? U(U(0) - U(v)) : U(v);
}

template <typename Coord>
auto image_power(unsigned at) -> image_pattern<Coord> {
  using U = image_pattern<Coord>;
  return U(U(1) << at);
}

template <typename Coord> auto image_highest() -> Coord {
  using U = image_pattern<Coord>;
  return image_valued<Coord>(U(U(~U(0)) >> 1));
}

template <typename Coord> auto image_lowest() -> Coord {
  return Coord(-image_highest<Coord>() - Coord(1));
}

/// The correctly rounded double of `v`: a magnitude of at most 53 bits is
/// the sum of its own powers, and a wider one is its top 53 bits carried by
/// `ldexp`, the bit below them deciding with what lies under it — above a
/// half ulp rounds away, exactly a half rounds to even.
template <typename Coord> auto image_rounded(const Coord &v) -> double {
  using U = image_pattern<Coord>;
  const U m = image_magnitude(v);
  const auto bit = [&m](unsigned k) { return U(U(m >> k) & U(1)) != U(0); };
  unsigned width = 0;
  for (unsigned k = 0; k < image_width<Coord>(); ++k)
    if (bit(k))
      width = k + 1;
  double magnitude = 0.0;
  if (width <= 53) {
    for (unsigned k = 0; k < width; ++k)
      if (bit(k))
        magnitude += std::ldexp(1.0, int(k));
  } else {
    const unsigned dropped = width - 53;
    std::uint64_t kept = 0;
    for (unsigned k = 0; k < 53; ++k)
      if (bit(dropped + k))
        kept |= std::uint64_t(1) << k;
    bool under = false;
    for (unsigned k = 0; k + 1 < dropped; ++k)
      under = under || bit(k);
    if (bit(dropped - 1))
      kept += under ? 1 : (kept & 1);
    magnitude = std::ldexp(double(kept), int(dropped));
  }
  return v < Coord(0) ? -magnitude : magnitude;
}

/// `a - b` past the type's bounds, from the bounds alone.
template <typename Coord>
auto image_overflows(const Coord &a, const Coord &b) -> bool {
  if (b < Coord(0))
    return a > Coord(image_highest<Coord>() + b);
  if (b > Coord(0))
    return a < Coord(image_lowest<Coord>() + b);
  return false;
}

template <typename Coord>
auto image_check(const Coord &a, const Coord &b) -> void {
  const double sentinel = -0.5;
  double image = sentinel;
  const auto kind = tf::exact::difference_image(a, b, image);
  CAPTURE(image_rounded(a), image_rounded(b));
  if (image_overflows(a, b)) {
    REQUIRE(kind == tf::exact::difference_image_kind::refused);
    REQUIRE(image == sentinel);
    return;
  }
  const Coord difference = Coord(a - b);
  const Coord limit = image_valued<Coord>(image_power<Coord>(52));
  const bool exact = difference > -limit && difference < limit;
  REQUIRE(kind == (exact ? tf::exact::difference_image_kind::exact
                         : tf::exact::difference_image_kind::rounded));
  REQUIRE(image == image_rounded(difference));
  if (exact)
    REQUIRE(std::abs(image) < std::ldexp(1.0, 52));
}

/// Words of the drawn bit width, negated on a drawn bit, so magnitudes at
/// every scale the type holds occur.
template <typename Coord, typename Engine>
auto image_draw(Engine &rng) -> Coord {
  using U = image_pattern<Coord>;
  constexpr unsigned bits = image_width<Coord>();
  const unsigned width = tf::test::uniform_int(rng, 0u, bits);
  if (width == 0)
    return Coord(0);
  U v = U(0);
  for (unsigned k = 0; k < bits; k += 64)
    v = U(v | U(U(tf::test::uniform_word(rng)) << k));
  v = U(U(v << (bits - width)) >> (bits - width));
  if (tf::test::uniform_coin(rng, 0.5))
    v = U(U(0) - v);
  return image_valued<Coord>(v);
}

/// The bounds, zero, one, the classification's own boundary and the widths
/// either side of it, each with its negation.
template <typename Coord> auto image_edges() -> std::vector<Coord> {
  using U = image_pattern<Coord>;
  std::vector<Coord> out{image_lowest<Coord>(), image_highest<Coord>(),
                         Coord(0)};
  for (const unsigned at :
       {0u, 51u, 52u, 53u, 63u, image_width<Coord>() - 1u}) {
    const U p = image_power<Coord>(at);
    for (const U v : {p, U(p - U(1)), U(p + U(1))}) {
      out.push_back(image_valued<Coord>(v));
      out.push_back(image_valued<Coord>(U(U(0) - v)));
    }
  }
  return out;
}

template <typename Coord> auto image_sweep(std::uint64_t seed) -> void {
  const auto edges = image_edges<Coord>();
  for (const Coord &a : edges)
    for (const Coord &b : edges)
      image_check(a, b);
  std::mt19937_64 rng(seed);
  for (std::size_t trial = 0; trial < 4096; ++trial)
    image_check(image_draw<Coord>(rng), image_draw<Coord>(rng));
}

template <typename Coord> auto image_states_its_verdicts() -> void {
  double image = 0.0;
  // The boundary the kind states: 2^52 - 1 is the difference itself, 2^52
  // is the difference rounded, either way round.
  const Coord limit = image_valued<Coord>(image_power<Coord>(52));
  REQUIRE(
      tf::exact::difference_image(Coord(limit - Coord(1)), Coord(0), image) ==
      tf::exact::difference_image_kind::exact);
  REQUIRE(image == std::ldexp(1.0, 52) - 1.0);
  REQUIRE(tf::exact::difference_image(limit, Coord(0), image) ==
          tf::exact::difference_image_kind::rounded);
  REQUIRE(image == std::ldexp(1.0, 52));
  REQUIRE(
      tf::exact::difference_image(Coord(0), Coord(limit - Coord(1)), image) ==
      tf::exact::difference_image_kind::exact);
  REQUIRE(image == 1.0 - std::ldexp(1.0, 52));
  REQUIRE(
      tf::exact::difference_image(Coord(-1), Coord(limit - Coord(1)), image) ==
      tf::exact::difference_image_kind::rounded);
  REQUIRE(image == -std::ldexp(1.0, 52));
  // The refusal forms no image.
  image = -0.5;
  REQUIRE(
      tf::exact::difference_image(image_highest<Coord>(), Coord(-1), image) ==
      tf::exact::difference_image_kind::refused);
  REQUIRE(image == -0.5);
  REQUIRE(tf::exact::difference_image(image_lowest<Coord>(), Coord(1), image) ==
          tf::exact::difference_image_kind::refused);
  REQUIRE(tf::exact::difference_image(image_highest<Coord>(),
                                      image_lowest<Coord>(), image) ==
          tf::exact::difference_image_kind::refused);
  // A difference that just fits is not one.
  REQUIRE(
      tf::exact::difference_image(image_highest<Coord>(), Coord(0), image) ==
      tf::exact::difference_image_kind::rounded);
  REQUIRE(image == std::ldexp(1.0, int(image_width<Coord>() - 1)));
}

} // namespace

TEST_CASE("difference_image: the three verdicts of a 64-bit difference",
          "[exact][difference_image]") {
  image_states_its_verdicts<tf::exact::int64>();
  image_sweep<tf::exact::int64>(0x6411e2026ull);
}

TEST_CASE("difference_image: the three verdicts of a 128-bit difference",
          "[exact][difference_image]") {
  image_states_its_verdicts<tf::exact::int128>();
  image_sweep<tf::exact::int128>(0x12811e2026ull);
}

/*
 * Copyright (c) 2026 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#pragma once
#include "./bits/twos_complement.hpp"
#include <cstdint>
#include <type_traits>

namespace tf::exact {

/// What a filter may assume of a difference's double image.
enum class difference_image_kind : std::uint8_t {
  refused, ///< No image was formed; the filter cannot decide.
  rounded, ///< The image is the difference rounded to nearest, its relative
           ///< error at most 2^-53.
  exact    ///< The image is the difference itself.
};

/// The double image of `a - b`, formed from the exact difference through
/// the coordinates' two's-complement pattern (@ref
/// tf::exact::bits::bit_pattern).
///
/// A built-in coordinate of at most 32 bits forms its difference in 64 bits,
/// where it is exact. Every wider one — 64-bit, and the 128-bit type whether
/// the compiler's own or the portable class — forms it wrap-free through its
/// unsigned pattern, refused where it overflows, and converts it rounded
/// once to nearest; below 2^52 in magnitude the image is the difference
/// itself, at and above it the difference rounded. Every route states the
/// same kind and the same image for the same pair of coordinates, so a
/// filter reading it decides the same way whichever formed it.
template <typename Coord>
auto difference_image(const Coord &a, const Coord &b, double &image)
    -> difference_image_kind {
  static_assert(bits::has_bit_pattern<Coord>,
                "a difference is formed through its two's-complement pattern");
  if constexpr (std::is_integral_v<Coord> &&
                sizeof(Coord) <= sizeof(std::int32_t)) {
    image = double(std::int64_t(a) - std::int64_t(b));
    return difference_image_kind::exact;
  } else {
    using U = typename bits::bit_pattern<Coord>::type;
    const Coord difference = bits::signed_value<Coord>(U(U(a) - U(b)));
    if (((a ^ b) & (a ^ difference)) < Coord(0))
      return difference_image_kind::refused;
    image = double(difference);
    const Coord limit = Coord(1) << 52;
    return difference > -limit && difference < limit
               ? difference_image_kind::exact
               : difference_image_kind::rounded;
  }
}

} // namespace tf::exact

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

/// The double image of `a - b`, formed from the exact difference.
///
/// A native coordinate of at most 64 bits forms its difference natively —
/// the 64-bit difference wrap-free through its unsigned pattern, refused
/// where it overflows — and converts it in hardware, which rounds to
/// nearest; below 2^52 in magnitude the image is the difference itself. A
/// wider coordinate forms it in `Wide`, which must hold every difference of
/// two of them, and states only an exact image. Every route states the same
/// image for the same difference, so a filter reading it decides the same
/// way whichever formed it.
template <typename Wide, typename Coord>
auto difference_image(const Coord &a, const Coord &b, double &image)
    -> difference_image_kind {
  if constexpr (std::is_integral_v<Coord> &&
                sizeof(Coord) <= sizeof(std::int32_t)) {
    image = double(std::int64_t(a) - std::int64_t(b));
    return difference_image_kind::exact;
  } else if constexpr (std::is_integral_v<Coord>) {
    using U = std::make_unsigned_t<Coord>;
    const Coord difference = bits::signed_value<Coord>(U(U(a) - U(b)));
    if (((a ^ b) & (a ^ difference)) < Coord(0))
      return difference_image_kind::refused;
    image = double(difference);
    const Coord limit = Coord(1) << 52;
    return difference > -limit && difference < limit
               ? difference_image_kind::exact
               : difference_image_kind::rounded;
  } else {
    const Wide difference = Wide(a) - Wide(b);
    const Wide limit = Wide(1) << 52;
    if (!(difference > -limit && difference < limit))
      return difference_image_kind::refused;
    image = double(difference);
    return difference_image_kind::exact;
  }
}

} // namespace tf::exact

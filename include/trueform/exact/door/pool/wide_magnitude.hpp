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

#include "../../int128.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::exact::door::pool {

/// How many product-rung magnitudes the widest comparison multiplies
/// together. The right side of the general branch is `(span B_i B_j)^2` —
/// six factors — and the left side's `C^2 H` is three, so six bounds the
/// whole predicate.
inline constexpr int wide_magnitude_factors = 6;

/// A NONNEGATIVE magnitude in fixed limb scratch, wide enough for the
/// squared-gap predicate and nothing else.
///
/// The predicate compares `C^2 H` against `(2T B_i B_j)^2`. On the int32
/// lattice the left side reaches 596 bits and the right 548, so no rung of
/// the library's ladder is a uniform carrier for it. What the comparison
/// needs is not a number type but the ORDER of two products of nonnegative
/// factors, which limb scratch and a schoolbook multiply state exactly with
/// no allocation and no new arithmetic tier.
///
/// THE CAPACITY IS THE LANE'S. Every factor the predicate feeds it is a
/// product-rung value, so `wide_magnitude_factors` of `ProductBits` bounds
/// every intermediate it forms: overflowing the scratch is impossible rather
/// than checked, and a wider lattice widens the scratch with the rung it
/// stands on.
template <int ProductBits> struct wide_magnitude {
  static constexpr std::size_t capacity =
      std::size_t(wide_magnitude_factors * ProductBits + 63) / 64;
  std::array<std::uint64_t, capacity> limb{};
  std::size_t used = 0;
};

/// The magnitude of a signed wide value, its limbs read off the halves the
/// type already stores.
template <typename Magnitude, typename Wide>
auto wide_magnitude_of(Wide value) -> Magnitude {
  Magnitude at;
  if (value < Wide(0))
    value = -value;
  for (std::size_t k = 0; k < Magnitude::capacity && value != Wide(0); ++k) {
    at.limb[k] = static_cast<std::uint64_t>(value);
    value = value >> 64;
    at.used = k + 1;
  }
  return at;
}

/// The product, exactly, in the same scratch. The capacity holds every
/// product the predicate forms, so the schoolbook walk needs no bound of
/// its own.
template <int ProductBits>
auto wide_multiply(const wide_magnitude<ProductBits> &a,
                   const wide_magnitude<ProductBits> &b)
    -> wide_magnitude<ProductBits> {
  wide_magnitude<ProductBits> at;
  if (a.used == 0 || b.used == 0)
    return at;
  for (std::size_t i = 0; i < a.used; ++i) {
    tf::exact::uint128 carry = 0;
    for (std::size_t j = 0; j < b.used; ++j) {
      const auto slot = i + j;
      const tf::exact::uint128 sum =
          tf::exact::uint128(at.limb[slot]) +
          tf::exact::uint128(a.limb[i]) * tf::exact::uint128(b.limb[j]) + carry;
      at.limb[slot] = static_cast<std::uint64_t>(sum);
      carry = sum >> 64;
    }
    for (std::size_t slot = i + b.used; carry != tf::exact::uint128(0);
         ++slot) {
      const tf::exact::uint128 sum = tf::exact::uint128(at.limb[slot]) + carry;
      at.limb[slot] = static_cast<std::uint64_t>(sum);
      carry = sum >> 64;
    }
  }
  at.used = 0;
  for (std::size_t k = wide_magnitude<ProductBits>::capacity; k > 0; --k)
    if (at.limb[k - 1] != 0) {
      at.used = k;
      break;
    }
  return at;
}

/// `-1`, `0` or `1` as `a` stands below, on or above `b`.
template <int ProductBits>
auto wide_compare(const wide_magnitude<ProductBits> &a,
                  const wide_magnitude<ProductBits> &b) -> int {
  if (a.used != b.used)
    return a.used < b.used ? -1 : 1;
  for (std::size_t k = a.used; k > 0; --k)
    if (a.limb[k - 1] != b.limb[k - 1])
      return a.limb[k - 1] < b.limb[k - 1] ? -1 : 1;
  return 0;
}

} // namespace tf::exact::door::pool

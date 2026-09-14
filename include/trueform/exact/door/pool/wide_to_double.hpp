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

#include <cmath>
#include <cstdint>

namespace tf::exact::door::pool {

/// The number of bits a machine word occupies.
inline auto wide_word_bits(std::uint64_t word) -> unsigned {
  unsigned bits = 0;
  if (word >> 32) {
    bits += 32;
    word >>= 32;
  }
  if (word >> 16) {
    bits += 16;
    word >>= 16;
  }
  if (word >> 8) {
    bits += 8;
    word >>= 8;
  }
  if (word >> 4) {
    bits += 4;
    word >>= 4;
  }
  if (word >> 2) {
    bits += 2;
    word >>= 2;
  }
  if (word >> 1) {
    bits += 1;
    word >>= 1;
  }
  return bits + unsigned(word);
}

/// The number of bits the magnitude of a wide signed value occupies.
template <typename Wide> auto wide_bit_width(Wide value) -> unsigned {
  if (value < Wide(0))
    value = -value;
  unsigned above = 0;
  while (value >> 64 != Wide(0)) {
    value = value >> 64;
    above += 64;
  }
  return above + wide_word_bits(static_cast<std::uint64_t>(value));
}

/// A wide signed value as a double, TRUNCATED TOWARD ZERO at 53 significant
/// bits, so the result is never further from the value than `2^-52` of it.
///
/// That bound is what lets a double screen decide an exact predicate
/// wherever the two sides stand further apart than the screen's own error,
/// and hand the rest to the exact comparison.
template <typename Wide> auto wide_to_double(Wide value) -> double {
  const bool negative = value < Wide(0);
  if (negative)
    value = -value;
  const unsigned bits = wide_bit_width(value);
  if (bits == 0)
    return 0.0;
  double at;
  if (bits <= 53)
    at = double(static_cast<std::uint64_t>(value));
  else {
    const unsigned drop = bits - 53;
    at = std::ldexp(double(static_cast<std::uint64_t>(value >> drop)),
                    int(drop));
  }
  return negative ? -at : at;
}

} // namespace tf::exact::door::pool

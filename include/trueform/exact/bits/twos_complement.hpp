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

#include "../int128.hpp"

#include <type_traits>

namespace tf::exact::bits {

/// The unsigned type a value's two's-complement pattern is carried in
/// while it is shifted or wrapped. Only the native integers have one:
/// their left shift of a negative value and their overflow are
/// undefined, so the pattern is moved unsigned and read back through
/// @ref signed_value. The class ladder types carry two's complement in
/// unsigned limbs already and are operated on directly.
template <typename T, typename = void> struct bit_pattern {};

template <typename T>
struct bit_pattern<T, std::enable_if_t<std::is_integral_v<T>>> {
  using type = std::make_unsigned_t<T>;
};

template <> struct bit_pattern<int128> {
  using type = uint128;
};

template <typename T, typename = void>
inline constexpr bool has_bit_pattern = false;

template <typename T>
inline constexpr bool
    has_bit_pattern<T, std::void_t<typename bit_pattern<T>::type>> = true;

/// The signed value of a two's-complement pattern. A set top bit is
/// read through the complement, which is always in range, so the
/// conversion back is never out of range.
template <typename T>
auto signed_value(const typename bit_pattern<T>::type &bits) -> T {
  using U = typename bit_pattern<T>::type;
  const unsigned width = unsigned(sizeof(T)) * 8u;
  if (U(bits >> (width - 1u)) == U(0))
    return static_cast<T>(bits);
  return T(-1) - static_cast<T>(U(~bits));
}

/// x * 2^w, exact while the product is representable, the wrapped
/// pattern past it. `w` is below the width of `T`.
template <typename T> auto shift_left(const T &x, unsigned w) -> T {
  if constexpr (has_bit_pattern<T>) {
    using U = typename bit_pattern<T>::type;
    return signed_value<T>(U(U(x) << w));
  } else {
    return x << w;
  }
}

/// floor(x / 2^w), the arithmetic shift right. `w` is below the width
/// of `T`.
template <typename T> auto shift_right(const T &x, unsigned w) -> T {
  if constexpr (has_bit_pattern<T>) {
    using U = typename bit_pattern<T>::type;
    const U ones = U(~U(0));
    const U fill = x < T(0) ? U(~U(ones >> w)) : U(0);
    return signed_value<T>(U(U(U(x) >> w) | fill));
  } else {
    return x >> w;
  }
}

/// floor(x / 2^w) * 2^w — the low `w` bits of the pattern cleared. The
/// result is always representable, so the pattern never wraps. `w` is
/// below the width of `T`.
template <typename T> auto align_down(const T &x, unsigned w) -> T {
  if constexpr (has_bit_pattern<T>) {
    using U = typename bit_pattern<T>::type;
    const U ones = U(~U(0));
    return signed_value<T>(U(U(x) & U(ones << w)));
  } else {
    return (x >> w) << w;
  }
}

/// a + b with the carry out of the top bit dropped — the wraparound an
/// accumulating kernel tracks with its own extension bit.
template <typename T> auto wrapping_add(const T &a, const T &b) -> T {
  if constexpr (has_bit_pattern<T>) {
    using U = typename bit_pattern<T>::type;
    return signed_value<T>(U(U(a) + U(b)));
  } else {
    return a + b;
  }
}

} // namespace tf::exact::bits

/*
 * Copyright (c) 2025 XLAB
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

#include "./int128.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace tf::exact {

/// Signed 256-bit integer. Two's complement, 2 x uint128 limbs (lo, hi).
/// Trivially default constructible (no zero-init).
class int256 {
public:
  using limb_type = tf::exact::uint128;

private:
  limb_type _lo;
  limb_type _hi;

  /// Full 128x128 -> 256 unsigned multiply. Returns low 128 bits, writes
  /// high 128 bits to @p hi_out.
  static auto _umul_full(limb_type a, limb_type b, limb_type &hi_out)
      -> limb_type {
    auto a0 = static_cast<std::uint64_t>(a);
    auto a1 = static_cast<std::uint64_t>(a >> 64);
    auto b0 = static_cast<std::uint64_t>(b);
    auto b1 = static_cast<std::uint64_t>(b >> 64);

    limb_type p00 = static_cast<limb_type>(a0) * b0;
    limb_type p01 = static_cast<limb_type>(a0) * b1;
    limb_type p10 = static_cast<limb_type>(a1) * b0;
    limb_type p11 = static_cast<limb_type>(a1) * b1;

    limb_type mid = (p00 >> 64) + static_cast<std::uint64_t>(p01) +
                    static_cast<std::uint64_t>(p10);
    hi_out = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    return (static_cast<limb_type>(static_cast<std::uint64_t>(p00))) |
           (mid << 64);
  }

  static constexpr auto _cmp_unsigned(const int256 &a, const int256 &b)
      -> int {
    if (a._hi != b._hi)
      return (a._hi < b._hi) ? -1 : 1;
    if (a._lo != b._lo)
      return (a._lo < b._lo) ? -1 : 1;
    return 0;
  }

  static auto _unsigned_add(const int256 &a, const int256 &b) -> int256 {
    limb_type lo = a._lo + b._lo;
    limb_type carry = (lo < a._lo) ? limb_type(1) : limb_type(0);
    limb_type hi = a._hi + b._hi + carry;
    return int256(lo, hi);
  }

  static auto _unsigned_sub(const int256 &a, const int256 &b) -> int256 {
    limb_type borrow = (a._lo < b._lo) ? limb_type(1) : limb_type(0);
    limb_type lo = a._lo - b._lo;
    limb_type hi = a._hi - b._hi - borrow;
    return int256(lo, hi);
  }

  static auto _unsigned_mul(const int256 &a, const int256 &b) -> int256 {
    limb_type lo_hi;
    limb_type lo_lo = _umul_full(a._lo, b._lo, lo_hi);
    limb_type cross = a._lo * b._hi + a._hi * b._lo;
    return int256(lo_lo, lo_hi + cross);
  }

  static auto _bit_width(const int256 &v) -> unsigned {
    if (v._hi != 0) {
      auto h = static_cast<std::uint64_t>(v._hi >> 64);
      if (h != 0) {
#if defined(__GNUC__) || defined(__clang__)
        return 192u + 64u - static_cast<unsigned>(__builtin_clzll(h));
#elif defined(_MSC_VER)
        unsigned long idx;
        _BitScanReverse64(&idx, h);
        return 192u + idx + 1u;
#else
        unsigned n = 0;
        while ((h >>= 1) != 0)
          ++n;
        return 192u + n + 1u;
#endif
      }
      auto l = static_cast<std::uint64_t>(v._hi);
#if defined(__GNUC__) || defined(__clang__)
      return 128u + 64u - static_cast<unsigned>(__builtin_clzll(l));
#elif defined(_MSC_VER)
      unsigned long idx;
      _BitScanReverse64(&idx, l);
      return 128u + idx + 1u;
#else
      unsigned n = 0;
      while ((l >>= 1) != 0)
        ++n;
      return 128u + n + 1u;
#endif
    }
    if (v._lo != 0) {
      auto h = static_cast<std::uint64_t>(v._lo >> 64);
      if (h != 0) {
#if defined(__GNUC__) || defined(__clang__)
        return 64u + 64u - static_cast<unsigned>(__builtin_clzll(h));
#elif defined(_MSC_VER)
        unsigned long idx;
        _BitScanReverse64(&idx, h);
        return 64u + idx + 1u;
#else
        unsigned n = 0;
        while ((h >>= 1) != 0)
          ++n;
        return 64u + n + 1u;
#endif
      }
      auto l = static_cast<std::uint64_t>(v._lo);
#if defined(__GNUC__) || defined(__clang__)
      return 64u - static_cast<unsigned>(__builtin_clzll(l));
#elif defined(_MSC_VER)
      unsigned long idx;
      _BitScanReverse64(&idx, l);
      return idx + 1u;
#else
      unsigned n = 0;
      while ((l >>= 1) != 0)
        ++n;
      return n + 1u;
#endif
    }
    return 0;
  }

  static constexpr auto _get_bit(const int256 &v, unsigned i) -> bool {
    if (i < 128)
      return ((v._lo >> i) & limb_type(1)) != 0;
    return ((v._hi >> (i - 128)) & limb_type(1)) != 0;
  }

  constexpr auto _set_bit(unsigned i) -> void {
    if (i < 128)
      _lo |= (limb_type(1) << i);
    else
      _hi |= (limb_type(1) << (i - 128));
  }

  static auto _unsigned_divmod(const int256 &num, const int256 &den)
      -> std::pair<int256, int256> {
    if (_cmp_unsigned(num, den) < 0)
      return {int256(0), num};

    int256 q(0);
    int256 r(0);
    const auto nbits = _bit_width(num);
    for (unsigned k = nbits; k-- > 0;) {
      r._hi = (r._hi << 1) | (r._lo >> 127);
      r._lo = (r._lo << 1) | static_cast<limb_type>(_get_bit(num, k));
      if (_cmp_unsigned(r, den) >= 0) {
        r = _unsigned_sub(r, den);
        q._set_bit(k);
      }
    }
    return {q, r};
  }

  auto _negate_in_place() -> void {
    _lo = ~_lo;
    _hi = ~_hi;
    _lo += 1;
    if (_lo == 0)
      _hi += 1;
  }

  static auto _abs(const int256 &v) -> int256 {
    if (!v.is_negative())
      return v;
    auto t = v;
    t._negate_in_place();
    return t;
  }

public:
  int256() = default;

  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  constexpr int256(T v) noexcept
      : _lo(static_cast<limb_type>(static_cast<tf::exact::int128>(v))),
        _hi(std::is_signed_v<T> && v < 0 ? ~limb_type(0) : limb_type(0)) {}

  constexpr int256(tf::exact::int128 v) noexcept
      : _lo(static_cast<limb_type>(v)),
        _hi(v < 0 ? ~limb_type(0) : limb_type(0)) {}

  constexpr explicit int256(limb_type lo, limb_type hi) noexcept
      : _lo(lo), _hi(hi) {}

  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  constexpr explicit operator T() const noexcept {
    return static_cast<T>(_lo);
  }

  constexpr explicit operator tf::exact::int128() const noexcept {
    return static_cast<tf::exact::int128>(_lo);
  }

  constexpr explicit operator bool() const noexcept {
    return (_lo | _hi) != 0;
  }

  [[nodiscard]] constexpr auto is_negative() const noexcept -> bool {
    return static_cast<tf::exact::int128>(_hi) < 0;
  }

  [[nodiscard]] constexpr auto is_zero() const noexcept -> bool {
    return (_lo | _hi) == 0;
  }

  [[nodiscard]] constexpr auto lo() const noexcept -> limb_type { return _lo; }
  [[nodiscard]] constexpr auto hi() const noexcept -> limb_type { return _hi; }

  // Full signed 256-bit value as double: split into two uint128 limbs (each
  // converted via uint128's operator double()), scaled by 2^128; negate to the
  // magnitude first and restore the sign.
  explicit operator double() const noexcept {
    constexpr double k_2pow128 =
        18446744073709551616.0 * 18446744073709551616.0; // (2^64)^2
    if (is_negative()) {
      const int256 m = -*this;
      return -(double(m.hi()) * k_2pow128 + double(m.lo()));
    }
    return double(hi()) * k_2pow128 + double(lo());
  }

  friend constexpr auto operator==(const int256 &a, const int256 &b) noexcept
      -> bool {
    return a._lo == b._lo && a._hi == b._hi;
  }

  friend constexpr auto operator!=(const int256 &a, const int256 &b) noexcept
      -> bool {
    return !(a == b);
  }

  friend constexpr auto operator<(const int256 &a, const int256 &b) noexcept
      -> bool {
    auto a_hi = static_cast<tf::exact::int128>(a._hi);
    auto b_hi = static_cast<tf::exact::int128>(b._hi);
    if (a_hi != b_hi)
      return a_hi < b_hi;
    return a._lo < b._lo;
  }

  friend constexpr auto operator>(const int256 &a, const int256 &b) noexcept
      -> bool {
    return b < a;
  }

  friend constexpr auto operator<=(const int256 &a, const int256 &b) noexcept
      -> bool {
    return !(b < a);
  }

  friend constexpr auto operator>=(const int256 &a, const int256 &b) noexcept
      -> bool {
    return !(a < b);
  }

  friend auto operator+(const int256 &a, const int256 &b) noexcept -> int256 {
    return _unsigned_add(a, b);
  }

  friend auto operator-(const int256 &a, const int256 &b) noexcept -> int256 {
    return _unsigned_sub(a, b);
  }

  friend auto operator-(const int256 &v) noexcept -> int256 {
    auto t = v;
    t._negate_in_place();
    return t;
  }

  friend auto operator*(const int256 &a, const int256 &b) noexcept -> int256 {
    return _unsigned_mul(a, b);
  }

  [[nodiscard]] friend auto divmod(const int256 &a, const int256 &b) noexcept
      -> std::pair<int256, int256> {
    const auto neg_q = a.is_negative() != b.is_negative();
    const auto neg_r = a.is_negative();
    auto [q, r] = _unsigned_divmod(_abs(a), _abs(b));
    if (neg_q)
      q._negate_in_place();
    if (neg_r && !r.is_zero())
      r._negate_in_place();
    return {q, r};
  }

  friend auto operator/(const int256 &a, const int256 &b) noexcept -> int256 {
    return divmod(a, b).first;
  }

  friend auto operator%(const int256 &a, const int256 &b) noexcept -> int256 {
    return divmod(a, b).second;
  }

  friend constexpr auto operator~(const int256 &v) noexcept -> int256 {
    return int256(~v._lo, ~v._hi);
  }

  friend constexpr auto operator&(const int256 &a, const int256 &b) noexcept
      -> int256 {
    return int256(a._lo & b._lo, a._hi & b._hi);
  }

  friend constexpr auto operator|(const int256 &a, const int256 &b) noexcept
      -> int256 {
    return int256(a._lo | b._lo, a._hi | b._hi);
  }

  friend constexpr auto operator^(const int256 &a, const int256 &b) noexcept
      -> int256 {
    return int256(a._lo ^ b._lo, a._hi ^ b._hi);
  }

  friend auto operator<<(const int256 &a, unsigned s) noexcept -> int256 {
    if (s >= 256)
      return int256(0);
    if (s == 0)
      return a;
    if (s >= 128)
      return int256(limb_type(0), a._lo << (s - 128));
    return int256(a._lo << s, (a._hi << s) | (a._lo >> (128 - s)));
  }

  friend auto operator>>(const int256 &a, unsigned s) noexcept -> int256 {
    if (s >= 256)
      return a.is_negative() ? int256(-1) : int256(0);
    if (s == 0)
      return a;
    auto fill = a.is_negative()
                    ? ~limb_type(0)
                    : limb_type(0);
    if (s >= 128) {
      auto signed_hi = static_cast<tf::exact::int128>(a._hi);
      return int256(
          static_cast<limb_type>(signed_hi >> (s - 128)), fill);
    }
    auto signed_hi = static_cast<tf::exact::int128>(a._hi);
    return int256((a._lo >> s) | (a._hi << (128 - s)),
                  static_cast<limb_type>(signed_hi >> s));
  }

  auto operator+=(const int256 &o) noexcept -> int256 & {
    return *this = *this + o;
  }
  auto operator-=(const int256 &o) noexcept -> int256 & {
    return *this = *this - o;
  }
  auto operator*=(const int256 &o) noexcept -> int256 & {
    return *this = *this * o;
  }
  auto operator/=(const int256 &o) noexcept -> int256 & {
    return *this = *this / o;
  }
  auto operator%=(const int256 &o) noexcept -> int256 & {
    return *this = *this % o;
  }
  auto operator&=(const int256 &o) noexcept -> int256 & {
    return *this = *this & o;
  }
  auto operator|=(const int256 &o) noexcept -> int256 & {
    return *this = *this | o;
  }
  auto operator^=(const int256 &o) noexcept -> int256 & {
    return *this = *this ^ o;
  }
  auto operator<<=(unsigned s) noexcept -> int256 & {
    return *this = *this << s;
  }
  auto operator>>=(unsigned s) noexcept -> int256 & {
    return *this = *this >> s;
  }

  auto operator++() noexcept -> int256 & { return *this += int256(1); }
  auto operator++(int) noexcept -> int256 {
    auto t = *this;
    ++(*this);
    return t;
  }
  auto operator--() noexcept -> int256 & { return *this -= int256(1); }
  auto operator--(int) noexcept -> int256 {
    auto t = *this;
    --(*this);
    return t;
  }
};

} // namespace tf::exact

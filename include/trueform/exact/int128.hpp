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

#include <cstdint>
#include <type_traits>
#include <utility>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace tf::exact {

#if defined(_MSC_VER)

class int128;

class uint128 {
public:
  using limb_type = std::uint64_t;

private:
  limb_type _lo = 0;
  limb_type _hi = 0;

  template <typename T> static constexpr auto _sign_fill(T v) noexcept {
    if constexpr (std::is_signed_v<T>)
      return v < 0 ? ~limb_type(0) : limb_type(0);
    else
      return limb_type(0);
  }

  static auto _umul64(limb_type a, limb_type b, limb_type &hi) noexcept
      -> limb_type {
#if defined(_M_X64)
    unsigned __int64 high;
    const auto low = _umul128(static_cast<unsigned __int64>(a),
                              static_cast<unsigned __int64>(b), &high);
    hi = static_cast<limb_type>(high);
    return static_cast<limb_type>(low);
#else
    const auto a0 = static_cast<limb_type>(static_cast<std::uint32_t>(a));
    const auto a1 = a >> 32;
    const auto b0 = static_cast<limb_type>(static_cast<std::uint32_t>(b));
    const auto b1 = b >> 32;
    const auto p00 = a0 * b0;
    const auto p01 = a0 * b1;
    const auto p10 = a1 * b0;
    const auto p11 = a1 * b1;
    const auto mid = (p00 >> 32) + static_cast<std::uint32_t>(p01) +
                     static_cast<std::uint32_t>(p10);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return (p00 & 0xffffffffull) | (mid << 32);
#endif
  }

  static constexpr auto _bit_width64(limb_type v) noexcept -> unsigned {
    unsigned n = 0;
    while (v != 0) {
      ++n;
      v >>= 1;
    }
    return n;
  }

  static constexpr auto _bit_width(const uint128 &v) noexcept -> unsigned {
    if (v._hi != 0)
      return 64u + _bit_width64(v._hi);
    return _bit_width64(v._lo);
  }

  static constexpr auto _get_bit(const uint128 &v, unsigned bit) noexcept
      -> bool {
    return bit < 64 ? ((v._lo >> bit) & 1u) != 0
                    : ((v._hi >> (bit - 64)) & 1u) != 0;
  }

  constexpr auto _set_bit(unsigned bit) noexcept -> void {
    if (bit < 64)
      _lo |= limb_type(1) << bit;
    else
      _hi |= limb_type(1) << (bit - 64);
  }

  static auto _divmod(uint128 num, uint128 den) noexcept
      -> std::pair<uint128, uint128> {
    if (num < den)
      return {uint128(0), num};

    uint128 q(0);
    uint128 r(0);
    const auto bits = _bit_width(num);
    for (unsigned bit = bits; bit-- > 0;) {
      r <<= 1;
      if (_get_bit(num, bit))
        r._lo |= 1;
      if (r >= den) {
        r -= den;
        q._set_bit(bit);
      }
    }
    return {q, r};
  }

public:
  constexpr uint128() noexcept = default;

  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  constexpr uint128(T v) noexcept
      : _lo(static_cast<limb_type>(v)), _hi(_sign_fill(v)) {}

  constexpr uint128(limb_type lo, limb_type hi) noexcept : _lo(lo), _hi(hi) {}
  constexpr uint128(int128 v) noexcept;

  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  constexpr explicit operator T() const noexcept {
    return static_cast<T>(_lo);
  }

  // Full 128-bit value as double via 64-bit limb split (the integral
  // operator above only keeps the low limb). Native __int128 on clang/gcc
  // converts directly; this matches it for the MSVC fallback class.
  explicit operator double() const noexcept {
    return double(_hi) * 18446744073709551616.0 /* 2^64 */ + double(_lo);
  }

  constexpr explicit operator bool() const noexcept {
    return (_lo | _hi) != 0;
  }

  constexpr explicit operator int128() const noexcept;

  [[nodiscard]] constexpr auto lo() const noexcept -> limb_type { return _lo; }
  [[nodiscard]] constexpr auto hi() const noexcept -> limb_type { return _hi; }

  friend constexpr auto operator==(uint128 a, uint128 b) noexcept -> bool {
    return a._lo == b._lo && a._hi == b._hi;
  }

  friend constexpr auto operator!=(uint128 a, uint128 b) noexcept -> bool {
    return !(a == b);
  }

  friend constexpr auto operator<(uint128 a, uint128 b) noexcept -> bool {
    return a._hi != b._hi ? a._hi < b._hi : a._lo < b._lo;
  }

  friend constexpr auto operator>(uint128 a, uint128 b) noexcept -> bool {
    return b < a;
  }

  friend constexpr auto operator<=(uint128 a, uint128 b) noexcept -> bool {
    return !(b < a);
  }

  friend constexpr auto operator>=(uint128 a, uint128 b) noexcept -> bool {
    return !(a < b);
  }

  friend constexpr auto operator+(uint128 v) noexcept -> uint128 { return v; }

  friend constexpr auto operator~(uint128 v) noexcept -> uint128 {
    return uint128(~v._lo, ~v._hi);
  }

  friend constexpr auto operator-(uint128 v) noexcept -> uint128 {
    v = ~v;
    ++v;
    return v;
  }

  friend constexpr auto operator+(uint128 a, uint128 b) noexcept -> uint128 {
    const auto lo = a._lo + b._lo;
    const auto carry = lo < a._lo ? limb_type(1) : limb_type(0);
    return uint128(lo, a._hi + b._hi + carry);
  }

  friend constexpr auto operator-(uint128 a, uint128 b) noexcept -> uint128 {
    const auto borrow = a._lo < b._lo ? limb_type(1) : limb_type(0);
    return uint128(a._lo - b._lo, a._hi - b._hi - borrow);
  }

  friend auto operator*(uint128 a, uint128 b) noexcept -> uint128 {
    limb_type hi;
    const auto lo = _umul64(a._lo, b._lo, hi);
    hi += a._lo * b._hi + a._hi * b._lo;
    return uint128(lo, hi);
  }

  friend auto operator/(uint128 a, uint128 b) noexcept -> uint128 {
    return _divmod(a, b).first;
  }

  friend auto operator%(uint128 a, uint128 b) noexcept -> uint128 {
    return _divmod(a, b).second;
  }

  friend constexpr auto operator&(uint128 a, uint128 b) noexcept -> uint128 {
    return uint128(a._lo & b._lo, a._hi & b._hi);
  }

  friend constexpr auto operator|(uint128 a, uint128 b) noexcept -> uint128 {
    return uint128(a._lo | b._lo, a._hi | b._hi);
  }

  friend constexpr auto operator^(uint128 a, uint128 b) noexcept -> uint128 {
    return uint128(a._lo ^ b._lo, a._hi ^ b._hi);
  }

  friend constexpr auto operator<<(uint128 a, unsigned s) noexcept -> uint128 {
    if (s >= 128)
      return uint128(0);
    if (s == 0)
      return a;
    if (s >= 64)
      return uint128(0, a._lo << (s - 64));
    return uint128(a._lo << s, (a._hi << s) | (a._lo >> (64 - s)));
  }

  friend constexpr auto operator>>(uint128 a, unsigned s) noexcept -> uint128 {
    if (s >= 128)
      return uint128(0);
    if (s == 0)
      return a;
    if (s >= 64)
      return uint128(a._hi >> (s - 64), 0);
    return uint128((a._lo >> s) | (a._hi << (64 - s)), a._hi >> s);
  }

  constexpr auto operator+=(uint128 v) noexcept -> uint128 & {
    return *this = *this + v;
  }
  constexpr auto operator-=(uint128 v) noexcept -> uint128 & {
    return *this = *this - v;
  }
  auto operator*=(uint128 v) noexcept -> uint128 & { return *this = *this * v; }
  auto operator/=(uint128 v) noexcept -> uint128 & { return *this = *this / v; }
  auto operator%=(uint128 v) noexcept -> uint128 & { return *this = *this % v; }
  constexpr auto operator&=(uint128 v) noexcept -> uint128 & {
    return *this = *this & v;
  }
  constexpr auto operator|=(uint128 v) noexcept -> uint128 & {
    return *this = *this | v;
  }
  constexpr auto operator^=(uint128 v) noexcept -> uint128 & {
    return *this = *this ^ v;
  }
  constexpr auto operator<<=(unsigned s) noexcept -> uint128 & {
    return *this = *this << s;
  }
  constexpr auto operator>>=(unsigned s) noexcept -> uint128 & {
    return *this = *this >> s;
  }

  constexpr auto operator++() noexcept -> uint128 & {
    *this += uint128(1);
    return *this;
  }
  constexpr auto operator++(int) noexcept -> uint128 {
    auto t = *this;
    ++(*this);
    return t;
  }
  constexpr auto operator--() noexcept -> uint128 & {
    *this -= uint128(1);
    return *this;
  }
  constexpr auto operator--(int) noexcept -> uint128 {
    auto t = *this;
    --(*this);
    return t;
  }
};

class int128 {
public:
  using limb_type = std::uint64_t;

private:
  limb_type _lo = 0;
  limb_type _hi = 0;

  static auto _umul64(limb_type a, limb_type b, limb_type &hi) noexcept
      -> limb_type {
#if defined(_M_X64)
    unsigned __int64 high;
    const auto low = _umul128(static_cast<unsigned __int64>(a),
                              static_cast<unsigned __int64>(b), &high);
    hi = static_cast<limb_type>(high);
    return static_cast<limb_type>(low);
#else
    const auto a0 = static_cast<limb_type>(static_cast<std::uint32_t>(a));
    const auto a1 = a >> 32;
    const auto b0 = static_cast<limb_type>(static_cast<std::uint32_t>(b));
    const auto b1 = b >> 32;
    const auto p00 = a0 * b0;
    const auto p01 = a0 * b1;
    const auto p10 = a1 * b0;
    const auto p11 = a1 * b1;
    const auto mid = (p00 >> 32) + static_cast<std::uint32_t>(p01) +
                     static_cast<std::uint32_t>(p10);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return (p00 & 0xffffffffull) | (mid << 32);
#endif
  }

  static auto _mul_i64_i64(std::int64_t a, std::int64_t b) noexcept -> int128 {
#if defined(_M_X64)
    __int64 high;
    const auto low =
        _mul128(static_cast<__int64>(a), static_cast<__int64>(b), &high);
    return int128(static_cast<limb_type>(low), static_cast<limb_type>(high));
#else
    return int128(uint128(int128(a)) * uint128(int128(b)));
#endif
  }

  static auto _mul_i128_i64(int128 a, std::int64_t b) noexcept -> int128 {
    const auto rhs_low = static_cast<limb_type>(b);
    const auto rhs_high = b < 0 ? ~limb_type(0) : limb_type(0);

    limb_type high_low;
    const auto low = _umul64(a._lo, rhs_low, high_low);
    const auto high = high_low + a._hi * rhs_low + a._lo * rhs_high;
    return int128(low, high);
  }

  template <typename T> static constexpr auto _sign_fill(T v) noexcept {
    if constexpr (std::is_signed_v<T>)
      return v < 0 ? ~limb_type(0) : limb_type(0);
    else
      return limb_type(0);
  }

  static constexpr auto _sar64(limb_type v, unsigned s) noexcept -> limb_type {
    if (s == 0)
      return v;
    const auto shifted = v >> s;
    if ((v & (limb_type(1) << 63)) == 0)
      return shifted;
    return shifted | (~limb_type(0) << (64 - s));
  }

  [[nodiscard]] constexpr auto _negative() const noexcept -> bool {
    return (_hi & (limb_type(1) << 63)) != 0;
  }

  [[nodiscard]] constexpr auto _fits_i64() const noexcept -> bool {
    const auto sign_fill =
        (_lo & (limb_type(1) << 63)) != 0 ? ~limb_type(0) : limb_type(0);
    return _hi == sign_fill;
  }

  [[nodiscard]] constexpr auto _as_i64() const noexcept -> std::int64_t {
    return static_cast<std::int64_t>(_lo);
  }

public:
  constexpr int128() noexcept = default;

  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  constexpr int128(T v) noexcept
      : _lo(static_cast<limb_type>(v)), _hi(_sign_fill(v)) {}

  constexpr int128(limb_type lo, limb_type hi) noexcept : _lo(lo), _hi(hi) {}
  constexpr int128(uint128 v) noexcept : _lo(v.lo()), _hi(v.hi()) {}

  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  constexpr explicit operator T() const noexcept {
    return static_cast<T>(_lo);
  }

  // Full signed 128-bit value as double: negate to the magnitude (exact
  // two's-complement negate), limb-split, restore the sign. Matches native
  // __int128's conversion used on clang/gcc.
  explicit operator double() const noexcept {
    if ((_hi >> 63) != 0) {
      const int128 m = -*this;
      return -(double(m._hi) * 18446744073709551616.0 /* 2^64 */ + double(m._lo));
    }
    return double(_hi) * 18446744073709551616.0 + double(_lo);
  }

  constexpr explicit operator bool() const noexcept {
    return (_lo | _hi) != 0;
  }

  constexpr explicit operator uint128() const noexcept {
    return uint128(_lo, _hi);
  }

  [[nodiscard]] constexpr auto lo() const noexcept -> limb_type { return _lo; }
  [[nodiscard]] constexpr auto hi() const noexcept -> limb_type { return _hi; }
  [[nodiscard]] constexpr auto is_negative() const noexcept -> bool {
    return _negative();
  }

  friend constexpr auto operator==(int128 a, int128 b) noexcept -> bool {
    return a._lo == b._lo && a._hi == b._hi;
  }

  friend constexpr auto operator!=(int128 a, int128 b) noexcept -> bool {
    return !(a == b);
  }

  friend constexpr auto operator<(int128 a, int128 b) noexcept -> bool {
    const auto a_neg = a._negative();
    const auto b_neg = b._negative();
    if (a_neg != b_neg)
      return a_neg;
    return a._hi != b._hi ? a._hi < b._hi : a._lo < b._lo;
  }

  friend constexpr auto operator>(int128 a, int128 b) noexcept -> bool {
    return b < a;
  }

  friend constexpr auto operator<=(int128 a, int128 b) noexcept -> bool {
    return !(b < a);
  }

  friend constexpr auto operator>=(int128 a, int128 b) noexcept -> bool {
    return !(a < b);
  }

  friend constexpr auto operator+(int128 v) noexcept -> int128 { return v; }

  friend constexpr auto operator~(int128 v) noexcept -> int128 {
    return int128(~v._lo, ~v._hi);
  }

  friend constexpr auto operator-(int128 v) noexcept -> int128 {
    v = ~v;
    ++v;
    return v;
  }

  friend constexpr auto operator+(int128 a, int128 b) noexcept -> int128 {
    const auto lo = a._lo + b._lo;
    const auto carry = lo < a._lo ? limb_type(1) : limb_type(0);
    return int128(lo, a._hi + b._hi + carry);
  }

  friend constexpr auto operator-(int128 a, int128 b) noexcept -> int128 {
    const auto borrow = a._lo < b._lo ? limb_type(1) : limb_type(0);
    return int128(a._lo - b._lo, a._hi - b._hi - borrow);
  }

  friend auto operator*(int128 a, int128 b) noexcept -> int128 {
    const auto a_fits_i64 = a._fits_i64();
    const auto b_fits_i64 = b._fits_i64();
    if (a_fits_i64 && b_fits_i64)
      return _mul_i64_i64(a._as_i64(), b._as_i64());
    if (b_fits_i64)
      return _mul_i128_i64(a, b._as_i64());
    if (a_fits_i64)
      return _mul_i128_i64(b, a._as_i64());
    return int128(uint128(a) * uint128(b));
  }

  friend auto operator/(int128 a, int128 b) noexcept -> int128 {
    const auto neg = a._negative() != b._negative();
    auto q = uint128(a._negative() ? uint128(-a) : uint128(a)) /
             uint128(b._negative() ? uint128(-b) : uint128(b));
    auto result = int128(q);
    return neg ? -result : result;
  }

  friend auto operator%(int128 a, int128 b) noexcept -> int128 {
    const auto neg = a._negative();
    const auto numerator = uint128(neg ? uint128(-a) : uint128(a));
    const auto denominator = uint128(b._negative() ? uint128(-b) : uint128(b));
    auto r = numerator % denominator;
    auto result = int128(r);
    return neg ? -result : result;
  }

  friend constexpr auto operator&(int128 a, int128 b) noexcept -> int128 {
    return int128(a._lo & b._lo, a._hi & b._hi);
  }

  friend constexpr auto operator|(int128 a, int128 b) noexcept -> int128 {
    return int128(a._lo | b._lo, a._hi | b._hi);
  }

  friend constexpr auto operator^(int128 a, int128 b) noexcept -> int128 {
    return int128(a._lo ^ b._lo, a._hi ^ b._hi);
  }

  friend constexpr auto operator<<(int128 a, unsigned s) noexcept -> int128 {
    return int128(uint128(a) << s);
  }

  friend constexpr auto operator>>(int128 a, unsigned s) noexcept -> int128 {
    if (s >= 128)
      return a._negative() ? int128(-1) : int128(0);
    if (s == 0)
      return a;
    const auto fill = a._negative() ? ~limb_type(0) : limb_type(0);
    if (s >= 64)
      return int128(_sar64(a._hi, s - 64), fill);
    return int128((a._lo >> s) | (a._hi << (64 - s)), _sar64(a._hi, s));
  }

  constexpr auto operator+=(int128 v) noexcept -> int128 & {
    return *this = *this + v;
  }
  constexpr auto operator-=(int128 v) noexcept -> int128 & {
    return *this = *this - v;
  }
  auto operator*=(int128 v) noexcept -> int128 & { return *this = *this * v; }
  auto operator/=(int128 v) noexcept -> int128 & { return *this = *this / v; }
  auto operator%=(int128 v) noexcept -> int128 & { return *this = *this % v; }
  constexpr auto operator&=(int128 v) noexcept -> int128 & {
    return *this = *this & v;
  }
  constexpr auto operator|=(int128 v) noexcept -> int128 & {
    return *this = *this | v;
  }
  constexpr auto operator^=(int128 v) noexcept -> int128 & {
    return *this = *this ^ v;
  }
  constexpr auto operator<<=(unsigned s) noexcept -> int128 & {
    return *this = *this << s;
  }
  constexpr auto operator>>=(unsigned s) noexcept -> int128 & {
    return *this = *this >> s;
  }

  constexpr auto operator++() noexcept -> int128 & {
    *this += int128(1);
    return *this;
  }
  constexpr auto operator++(int) noexcept -> int128 {
    auto t = *this;
    ++(*this);
    return t;
  }
  constexpr auto operator--() noexcept -> int128 & {
    *this -= int128(1);
    return *this;
  }
  constexpr auto operator--(int) noexcept -> int128 {
    auto t = *this;
    --(*this);
    return t;
  }
};

constexpr uint128::uint128(int128 v) noexcept : _lo(v.lo()), _hi(v.hi()) {}

constexpr uint128::operator int128() const noexcept {
  return int128(_lo, _hi);
}

#elif defined(__GNUC__) || defined(__clang__)
using int128 = __int128;
using uint128 = unsigned __int128;
#else
#error "No 128-bit integer type available for this compiler"
#endif

} // namespace tf::exact

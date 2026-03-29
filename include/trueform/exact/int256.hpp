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

/// Signed 256-bit integer. Two's complement, 4 x uint64 limbs (low to high).
/// Trivially default constructible (no zero-init).
class int256 {
public:
  using limb_type = std::uint64_t;

private:
  limb_type _data[4];

#if defined(_MSC_VER) && defined(_M_X64)
  static inline auto _addcarry(limb_type a, limb_type b, limb_type cin,
                               limb_type &out) -> limb_type {
    return static_cast<limb_type>(
        _addcarry_u64(static_cast<unsigned char>(cin), a, b, &out));
  }

  static inline auto _subborrow(limb_type a, limb_type b, limb_type bin,
                                limb_type &out) -> limb_type {
    return static_cast<limb_type>(
        _subborrow_u64(static_cast<unsigned char>(bin), a, b, &out));
  }

  static inline auto _mul128(limb_type a, limb_type b, limb_type &hi)
      -> limb_type {
    return _umul128(a, b, &hi);
  }
#elif defined(__SIZEOF_INT128__)
  static inline auto _addcarry(limb_type a, limb_type b, limb_type cin,
                               limb_type &out) -> limb_type {
    using u128 = unsigned __int128;
    const auto s =
        static_cast<u128>(a) + static_cast<u128>(b) + static_cast<u128>(cin);
    out = static_cast<limb_type>(s);
    return static_cast<limb_type>(s >> 64);
  }

  static inline auto _subborrow(limb_type a, limb_type b, limb_type bin,
                                limb_type &out) -> limb_type {
    out = a - b - bin;
    return static_cast<limb_type>((a < b) || (bin && a == b));
  }

  static inline auto _mul128(limb_type a, limb_type b, limb_type &hi)
      -> limb_type {
    using u128 = unsigned __int128;
    const auto p = static_cast<u128>(a) * static_cast<u128>(b);
    hi = static_cast<limb_type>(p >> 64);
    return static_cast<limb_type>(p);
  }
#else
  static inline auto _addcarry(limb_type a, limb_type b, limb_type cin,
                               limb_type &out) -> limb_type {
    const auto s1 = static_cast<limb_type>(a + b);
    const auto c1 = static_cast<limb_type>(s1 < a);
    const auto s2 = static_cast<limb_type>(s1 + cin);
    const auto c2 = static_cast<limb_type>(s2 < s1);
    out = s2;
    return static_cast<limb_type>(c1 + c2);
  }

  static inline auto _subborrow(limb_type a, limb_type b, limb_type bin,
                                limb_type &out) -> limb_type {
    const auto t1 = static_cast<limb_type>(a - b);
    const auto b1 = static_cast<limb_type>(a < b);
    const auto t2 = static_cast<limb_type>(t1 - bin);
    const auto b2 = static_cast<limb_type>(t1 < bin);
    out = t2;
    return static_cast<limb_type>(b1 + b2);
  }

  static inline auto _mul128(limb_type a, limb_type b, limb_type &hi)
      -> limb_type {
    const auto a0 = static_cast<limb_type>(static_cast<std::uint32_t>(a));
    const auto a1 = static_cast<limb_type>(a >> 32);
    const auto b0 = static_cast<limb_type>(static_cast<std::uint32_t>(b));
    const auto b1 = static_cast<limb_type>(b >> 32);

    const auto p00 = a0 * b0;
    const auto p01 = a0 * b1;
    const auto p10 = a1 * b0;
    const auto p11 = a1 * b1;

    const auto mid = (p00 >> 32) + static_cast<std::uint32_t>(p01) +
                     static_cast<std::uint32_t>(p10);

    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return (p00 & 0xffffffffull) | (mid << 32);
  }
#endif

  static constexpr auto _cmp_unsigned(const int256 &a, const int256 &b)
      -> int {
    for (int i = 3; i >= 0; --i) {
      if (a._data[i] < b._data[i])
        return -1;
      if (a._data[i] > b._data[i])
        return 1;
    }
    return 0;
  }

  static auto _unsigned_add(const int256 &a, const int256 &b) -> int256 {
    int256 r;
    limb_type c = 0;
    c = _addcarry(a._data[0], b._data[0], c, r._data[0]);
    c = _addcarry(a._data[1], b._data[1], c, r._data[1]);
    c = _addcarry(a._data[2], b._data[2], c, r._data[2]);
    (void)_addcarry(a._data[3], b._data[3], c, r._data[3]);
    return r;
  }

  static auto _unsigned_sub(const int256 &a, const int256 &b) -> int256 {
    int256 r;
    limb_type c = 0;
    c = _subborrow(a._data[0], b._data[0], c, r._data[0]);
    c = _subborrow(a._data[1], b._data[1], c, r._data[1]);
    c = _subborrow(a._data[2], b._data[2], c, r._data[2]);
    (void)_subborrow(a._data[3], b._data[3], c, r._data[3]);
    return r;
  }

  static auto _unsigned_mul(const int256 &a, const int256 &b) -> int256 {
    int256 r{};
    auto add_at = [&](unsigned idx, limb_type lo, limb_type hi) {
      limb_type carry = 0;
      carry = _addcarry(r._data[idx], lo, 0, r._data[idx]);
      if (idx + 1 < 4) {
        carry = _addcarry(r._data[idx + 1], hi, carry, r._data[idx + 1]);
        for (unsigned k = idx + 2; k < 4 && carry; ++k)
          carry = _addcarry(r._data[k], 0, carry, r._data[k]);
      }
    };
    for (unsigned i = 0; i < 4; ++i)
      for (unsigned j = 0; i + j < 4; ++j) {
        limb_type hi = 0;
        const auto lo = _mul128(a._data[i], b._data[j], hi);
        add_at(i + j, lo, hi);
      }
    return r;
  }

  static auto _bit_width(const int256 &v) -> unsigned {
    for (int i = 3; i >= 0; --i) {
      if (v._data[i] != 0) {
#if defined(__GNUC__) || defined(__clang__)
        return static_cast<unsigned>(i) * 64u + 64u -
               static_cast<unsigned>(__builtin_clzll(v._data[i]));
#elif defined(_MSC_VER)
        unsigned long idx;
        _BitScanReverse64(&idx, v._data[i]);
        return static_cast<unsigned>(i) * 64u + idx + 1u;
#else
        unsigned n = 0;
        auto t = v._data[i];
        while ((t >>= 1) != 0)
          ++n;
        return static_cast<unsigned>(i) * 64u + n + 1u;
#endif
      }
    }
    return 0;
  }

  static constexpr auto _get_bit(const int256 &v, unsigned i) -> bool {
    return ((v._data[i / 64u] >> (i % 64u)) & 1ull) != 0;
  }

  constexpr auto _set_bit(unsigned i) -> void {
    _data[i / 64u] |= (1ull << (i % 64u));
  }

  static auto _unsigned_divmod(const int256 &num, const int256 &den)
      -> std::pair<int256, int256> {
    if (_cmp_unsigned(num, den) < 0)
      return {int256(0), num};

    int256 q(0);
    int256 r(0);
    const auto nbits = _bit_width(num);
    for (unsigned k = nbits; k-- > 0;) {
      r._data[3] = (r._data[3] << 1) | (r._data[2] >> 63);
      r._data[2] = (r._data[2] << 1) | (r._data[1] >> 63);
      r._data[1] = (r._data[1] << 1) | (r._data[0] >> 63);
      r._data[0] = (r._data[0] << 1) |
                    static_cast<limb_type>(_get_bit(num, k));
      if (_cmp_unsigned(r, den) >= 0) {
        r = _unsigned_sub(r, den);
        q._set_bit(k);
      }
    }
    return {q, r};
  }

  auto _negate_in_place() -> void {
    _data[0] = ~_data[0];
    _data[1] = ~_data[1];
    _data[2] = ~_data[2];
    _data[3] = ~_data[3];
    limb_type c = 1;
    c = _addcarry(_data[0], 0, c, _data[0]);
    c = _addcarry(_data[1], 0, c, _data[1]);
    c = _addcarry(_data[2], 0, c, _data[2]);
    (void)_addcarry(_data[3], 0, c, _data[3]);
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
  constexpr int256(T v) noexcept : _data{static_cast<limb_type>(v), 0, 0, 0} {
    if constexpr (std::is_signed_v<T>) {
      if (v < 0) {
        _data[1] = ~limb_type(0);
        _data[2] = ~limb_type(0);
        _data[3] = ~limb_type(0);
      }
    }
  }

  constexpr int256(tf::exact::int128 v) noexcept
      : _data{static_cast<limb_type>(v),
              static_cast<limb_type>(v >> 64), 0, 0} {
    if (v < 0) {
      _data[2] = ~limb_type(0);
      _data[3] = ~limb_type(0);
    }
  }

  constexpr explicit int256(limb_type x0, limb_type x1, limb_type x2,
                            limb_type x3) noexcept
      : _data{x0, x1, x2, x3} {}

  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  constexpr explicit operator T() const noexcept {
    return static_cast<T>(_data[0]);
  }

  constexpr explicit operator bool() const noexcept {
    return (_data[0] | _data[1] | _data[2] | _data[3]) != 0;
  }

  [[nodiscard]] constexpr auto is_negative() const noexcept -> bool {
    return (_data[3] >> 63) != 0;
  }

  [[nodiscard]] constexpr auto is_zero() const noexcept -> bool {
    return (_data[0] | _data[1] | _data[2] | _data[3]) == 0;
  }

  [[nodiscard]] constexpr auto raw_limb(unsigned i) const noexcept
      -> limb_type {
    return _data[i];
  }

  friend constexpr auto operator==(const int256 &a, const int256 &b) noexcept
      -> bool {
    return a._data[0] == b._data[0] && a._data[1] == b._data[1] &&
           a._data[2] == b._data[2] && a._data[3] == b._data[3];
  }

  friend constexpr auto operator!=(const int256 &a, const int256 &b) noexcept
      -> bool {
    return !(a == b);
  }

  friend constexpr auto operator<(const int256 &a, const int256 &b) noexcept
      -> bool {
    auto a3 = static_cast<std::int64_t>(a._data[3]);
    auto b3 = static_cast<std::int64_t>(b._data[3]);
    if (a3 != b3)
      return a3 < b3;
    if (a._data[2] != b._data[2])
      return a._data[2] < b._data[2];
    if (a._data[1] != b._data[1])
      return a._data[1] < b._data[1];
    return a._data[0] < b._data[0];
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
    return int256(~v._data[0], ~v._data[1], ~v._data[2], ~v._data[3]);
  }

  friend constexpr auto operator&(const int256 &a, const int256 &b) noexcept
      -> int256 {
    return int256(a._data[0] & b._data[0], a._data[1] & b._data[1],
                  a._data[2] & b._data[2], a._data[3] & b._data[3]);
  }

  friend constexpr auto operator|(const int256 &a, const int256 &b) noexcept
      -> int256 {
    return int256(a._data[0] | b._data[0], a._data[1] | b._data[1],
                  a._data[2] | b._data[2], a._data[3] | b._data[3]);
  }

  friend constexpr auto operator^(const int256 &a, const int256 &b) noexcept
      -> int256 {
    return int256(a._data[0] ^ b._data[0], a._data[1] ^ b._data[1],
                  a._data[2] ^ b._data[2], a._data[3] ^ b._data[3]);
  }

  friend auto operator<<(const int256 &a, unsigned s) noexcept -> int256 {
    if (s >= 256)
      return int256(0);
    if (s == 0)
      return a;
    int256 r;
    const auto ls = s / 64u;
    const auto bs = s % 64u;
    for (unsigned i = 0; i < 4; ++i) {
      if (i < ls) {
        r._data[i] = 0;
        continue;
      }
      auto src = i - ls;
      auto v = a._data[src] << bs;
      if (bs != 0 && src > 0)
        v |= a._data[src - 1] >> (64u - bs);
      r._data[i] = v;
    }
    return r;
  }

  friend auto operator>>(const int256 &a, unsigned s) noexcept -> int256 {
    if (s >= 256)
      return a.is_negative() ? int256(-1) : int256(0);
    if (s == 0)
      return a;
    int256 r;
    const auto ls = s / 64u;
    const auto bs = s % 64u;
    const auto fill = a.is_negative() ? ~limb_type(0) : limb_type(0);
    for (unsigned i = 0; i < 4; ++i) {
      auto src = i + ls;
      if (src >= 4) {
        r._data[i] = fill;
        continue;
      }
      auto v = a._data[src] >> bs;
      if (bs != 0) {
        auto hi = (src + 1 < 4) ? a._data[src + 1] : fill;
        v |= hi << (64u - bs);
      }
      r._data[i] = v;
    }
    return r;
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

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
#include "./int256.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace tf::exact {

/// Signed 512-bit integer. Two's complement, 4 x uint128 limbs, least
/// significant first. Trivially default constructible (no zero-init).
class int512 {
public:
  using limb_type = tf::exact::uint128;

private:
  limb_type _limb[4];

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

  static constexpr auto _of_limbs(limb_type l0, limb_type l1, limb_type l2,
                                  limb_type l3) -> int512 {
    return int512(l0, l1, l2, l3);
  }

  static constexpr auto _cmp_unsigned(const int512 &a, const int512 &b)
      -> int {
    for (std::size_t k = 4; k-- > 0;)
      if (a._limb[k] != b._limb[k])
        return (a._limb[k] < b._limb[k]) ? -1 : 1;
    return 0;
  }

  static auto _unsigned_add(const int512 &a, const int512 &b) -> int512 {
    limb_type out[4];
    limb_type carry(0);
    for (std::size_t k = 0; k < 4; ++k) {
      const limb_type sum = a._limb[k] + b._limb[k];
      const limb_type spill = (sum < a._limb[k]) ? limb_type(1) : limb_type(0);
      const limb_type total = sum + carry;
      carry = spill | ((total < sum) ? limb_type(1) : limb_type(0));
      out[k] = total;
    }
    return _of_limbs(out[0], out[1], out[2], out[3]);
  }

  static auto _unsigned_sub(const int512 &a, const int512 &b) -> int512 {
    limb_type out[4];
    limb_type borrow(0);
    for (std::size_t k = 0; k < 4; ++k) {
      const limb_type diff = a._limb[k] - b._limb[k];
      const limb_type under =
          (a._limb[k] < b._limb[k]) ? limb_type(1) : limb_type(0);
      const limb_type total = diff - borrow;
      borrow = under | ((diff < borrow) ? limb_type(1) : limb_type(0));
      out[k] = total;
    }
    return _of_limbs(out[0], out[1], out[2], out[3]);
  }

  static auto _accumulate(limb_type &slot, limb_type value, limb_type &carry)
      -> void {
    slot += value;
    carry += (slot < value) ? limb_type(1) : limb_type(0);
  }

  /// The partial products are formed first and combined after, so the
  /// multiplies do not sit behind one another's carries. The top slot needs
  /// no high half — the product is truncating — so only six of the ten
  /// partials are widened.
  static auto _unsigned_mul(const int512 &a, const int512 &b) -> int512 {
    limb_type h00, h01, h10, h11, h02, h20;
    const limb_type l00 = _umul_full(a._limb[0], b._limb[0], h00);
    const limb_type l01 = _umul_full(a._limb[0], b._limb[1], h01);
    const limb_type l10 = _umul_full(a._limb[1], b._limb[0], h10);
    const limb_type l11 = _umul_full(a._limb[1], b._limb[1], h11);
    const limb_type l02 = _umul_full(a._limb[0], b._limb[2], h02);
    const limb_type l20 = _umul_full(a._limb[2], b._limb[0], h20);

    limb_type slot1 = h00, carry1(0);
    _accumulate(slot1, l01, carry1);
    _accumulate(slot1, l10, carry1);

    limb_type slot2 = h01, carry2(0);
    _accumulate(slot2, h10, carry2);
    _accumulate(slot2, l11, carry2);
    _accumulate(slot2, l02, carry2);
    _accumulate(slot2, l20, carry2);
    _accumulate(slot2, carry1, carry2);

    const limb_type slot3 = h11 + h02 + h20 + carry2 +
                            a._limb[0] * b._limb[3] +
                            a._limb[1] * b._limb[2] +
                            a._limb[2] * b._limb[1] + a._limb[3] * b._limb[0];

    return _of_limbs(l00, slot1, slot2, slot3);
  }

  static auto _word_bits(std::uint64_t word) -> unsigned {
#if defined(__GNUC__) || defined(__clang__)
    return 64u - static_cast<unsigned>(__builtin_clzll(word));
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    unsigned long index;
    _BitScanReverse64(&index, word);
    return static_cast<unsigned>(index) + 1u;
#else
    unsigned n = 0;
    while ((word >>= 1) != 0)
      ++n;
    return n + 1u;
#endif
  }

  static auto _bit_width(const int512 &v) -> unsigned {
    for (std::size_t k = 4; k-- > 0;) {
      const auto high = static_cast<std::uint64_t>(v._limb[k] >> 64);
      if (high != 0)
        return unsigned(k) * 128u + 64u + _word_bits(high);
      const auto low = static_cast<std::uint64_t>(v._limb[k]);
      if (low != 0)
        return unsigned(k) * 128u + _word_bits(low);
    }
    return 0;
  }

  /// `_unsigned_divmod` works on MAGNITUDES, and one magnitude — 2^511,
  /// what `_abs` returns for the most negative value — carries the sign
  /// bit; the signed shift would sign-extend it, collapse the
  /// quotient-digit estimate to zero, and leave the exact-correction loop
  /// walking the quotient one divisor at a time.
  static auto _logical_shr(const int512 &a, unsigned s) -> int512 {
    if (s >= 512)
      return int512(0);
    if (s == 0)
      return a;
    const std::size_t words = s / 128;
    const unsigned bits = s % 128;
    limb_type out[4] = {limb_type(0), limb_type(0), limb_type(0),
                        limb_type(0)};
    for (std::size_t k = 0; k + words < 4; ++k) {
      const std::size_t source = k + words;
      limb_type v = a._limb[source] >> bits;
      if (bits != 0 && source + 1 < 4)
        v |= a._limb[source + 1] << (128 - bits);
      out[k] = v;
    }
    return _of_limbs(out[0], out[1], out[2], out[3]);
  }

  /// 512 x 64 -> 512 unsigned multiply; the caller guarantees the product
  /// fits (every use multiplies a quotient-digit estimate that is bounded by
  /// a remainder already held in 512 bits).
  static auto _mul_small(const int512 &a, std::uint64_t b) -> int512 {
    const limb_type m(b);
    limb_type out[4];
    limb_type carry(0);
    for (std::size_t k = 0; k < 4; ++k) {
      limb_type high;
      const limb_type low = _umul_full(a._limb[k], m, high);
      out[k] = low + carry;
      carry = high + ((out[k] < carry) ? limb_type(1) : limb_type(0));
    }
    return _of_limbs(out[0], out[1], out[2], out[3]);
  }

  static auto _unsigned_divmod(const int512 &num, const int512 &den)
      -> std::pair<int512, int512> {
    if (_cmp_unsigned(num, den) < 0)
      return {int512(0), num};

    const auto nbits = _bit_width(num);
    const auto dbits = _bit_width(den);

    // narrow divisor: schoolbook over 64-bit digits, hardware-backed
    // 128/64 division per digit
    if (dbits <= 64) {
      const auto d = static_cast<std::uint64_t>(den._limb[0]);
      std::uint64_t digits[8];
      for (std::size_t k = 0; k < 4; ++k) {
        digits[2 * k] = static_cast<std::uint64_t>(num._limb[k]);
        digits[2 * k + 1] = static_cast<std::uint64_t>(num._limb[k] >> 64);
      }
      limb_type rem(0);
      for (std::size_t i = 8; i-- > 0;) {
        const limb_type cur = (rem << 64) | limb_type(digits[i]);
        digits[i] = static_cast<std::uint64_t>(cur / d);
        rem = cur % d;
      }
      limb_type out[4];
      for (std::size_t k = 0; k < 4; ++k)
        out[k] =
            limb_type(digits[2 * k]) | (limb_type(digits[2 * k + 1]) << 64);
      return {_of_limbs(out[0], out[1], out[2], out[3]),
              _of_limbs(rem, limb_type(0), limb_type(0), limb_type(0))};
    }

    // wide divisor: 32-bit quotient digits, each estimated from the
    // remainder's and divisor's top bits with an underestimating
    // denominator, then corrected exactly — the estimate is at most a
    // few below the true digit, so the correction loop is O(1).
    // The chunk count's own margin keeps the shifted remainder inside the
    // width: the first partial dividend holds at most dbits - 32 bits, so a
    // later round's `r << 32` needs dbits + 32, and a third round exists
    // only when nbits is at least dbits + 33.
    const unsigned s = dbits - 64;
    const auto den_top = _logical_shr(den, s)._limb[0] + limb_type(1);
    const unsigned n_chunks = (nbits - dbits + 32 + 31) / 32;
    int512 q(0);
    int512 r = _logical_shr(num, 32 * n_chunks);
    for (unsigned c = n_chunks; c-- > 0;) {
      const auto chunk =
          static_cast<std::uint64_t>(_logical_shr(num, 32 * c)._limb[0]) &
          0xffffffffull;
      r = _unsigned_add(r << 32, int512(chunk));
      const limb_type r_top = _logical_shr(r, s)._limb[0];
      auto digit = static_cast<std::uint64_t>(r_top / den_top);
      r = _unsigned_sub(r, _mul_small(den, digit));
      while (_cmp_unsigned(r, den) >= 0) {
        r = _unsigned_sub(r, den);
        ++digit;
      }
      q = _unsigned_add(q << 32, int512(digit));
    }
    return {q, r};
  }

  auto _negate_in_place() -> void {
    limb_type carry(1);
    for (std::size_t k = 0; k < 4; ++k) {
      _limb[k] = ~_limb[k] + carry;
      carry = (carry != limb_type(0) && _limb[k] == limb_type(0))
                  ? limb_type(1)
                  : limb_type(0);
    }
  }

  static auto _abs(const int512 &v) -> int512 {
    if (!v.is_negative())
      return v;
    auto t = v;
    t._negate_in_place();
    return t;
  }

public:
  int512() = default;

  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  constexpr int512(T v) noexcept
      : _limb{static_cast<limb_type>(static_cast<tf::exact::int128>(v)),
              std::is_signed_v<T> && v < 0 ? ~limb_type(0) : limb_type(0),
              std::is_signed_v<T> && v < 0 ? ~limb_type(0) : limb_type(0),
              std::is_signed_v<T> && v < 0 ? ~limb_type(0) : limb_type(0)} {}

  constexpr int512(tf::exact::int128 v) noexcept
      : _limb{static_cast<limb_type>(v), v < 0 ? ~limb_type(0) : limb_type(0),
              v < 0 ? ~limb_type(0) : limb_type(0),
              v < 0 ? ~limb_type(0) : limb_type(0)} {}

  constexpr int512(tf::exact::int256 v) noexcept
      : _limb{v.lo(), v.hi(),
              v.is_negative() ? ~limb_type(0) : limb_type(0),
              v.is_negative() ? ~limb_type(0) : limb_type(0)} {}

  constexpr explicit int512(limb_type l0, limb_type l1, limb_type l2,
                            limb_type l3) noexcept
      : _limb{l0, l1, l2, l3} {}

  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  constexpr explicit operator T() const noexcept {
    return static_cast<T>(_limb[0]);
  }

  constexpr explicit operator tf::exact::int128() const noexcept {
    return static_cast<tf::exact::int128>(_limb[0]);
  }

  constexpr explicit operator tf::exact::int256() const noexcept {
    return tf::exact::int256(_limb[0], _limb[1]);
  }

  constexpr explicit operator bool() const noexcept { return !is_zero(); }

  [[nodiscard]] constexpr auto is_negative() const noexcept -> bool {
    return static_cast<tf::exact::int128>(_limb[3]) < 0;
  }

  [[nodiscard]] constexpr auto is_zero() const noexcept -> bool {
    return (_limb[0] | _limb[1] | _limb[2] | _limb[3]) == 0;
  }

  [[nodiscard]] constexpr auto limb(std::size_t k) const noexcept
      -> limb_type {
    return _limb[k];
  }

  // Full signed 512-bit value as double: negate to the magnitude, read each
  // half as the unsigned pair of limbs it is, and restore the sign. The
  // halves are read through their limbs rather than through int256, whose
  // own conversion is signed and would call a low half negative.
  explicit operator double() const noexcept {
    constexpr double k_2pow64 = 18446744073709551616.0;
    const double k_2pow128 = k_2pow64 * k_2pow64;
    const double k_2pow256 = k_2pow128 * k_2pow128;
    const int512 magnitude = is_negative() ? -*this : *this;
    const double low = double(magnitude._limb[1]) * k_2pow128 +
                       double(magnitude._limb[0]);
    const double high = double(magnitude._limb[3]) * k_2pow128 +
                        double(magnitude._limb[2]);
    const double at = high * k_2pow256 + low;
    return is_negative() ? -at : at;
  }

  friend constexpr auto operator==(const int512 &a, const int512 &b) noexcept
      -> bool {
    return a._limb[0] == b._limb[0] && a._limb[1] == b._limb[1] &&
           a._limb[2] == b._limb[2] && a._limb[3] == b._limb[3];
  }

  friend constexpr auto operator!=(const int512 &a, const int512 &b) noexcept
      -> bool {
    return !(a == b);
  }

  friend constexpr auto operator<(const int512 &a, const int512 &b) noexcept
      -> bool {
    const auto a_top = static_cast<tf::exact::int128>(a._limb[3]);
    const auto b_top = static_cast<tf::exact::int128>(b._limb[3]);
    if (a_top != b_top)
      return a_top < b_top;
    for (std::size_t k = 3; k-- > 0;)
      if (a._limb[k] != b._limb[k])
        return a._limb[k] < b._limb[k];
    return false;
  }

  friend constexpr auto operator>(const int512 &a, const int512 &b) noexcept
      -> bool {
    return b < a;
  }

  friend constexpr auto operator<=(const int512 &a, const int512 &b) noexcept
      -> bool {
    return !(b < a);
  }

  friend constexpr auto operator>=(const int512 &a, const int512 &b) noexcept
      -> bool {
    return !(a < b);
  }

  friend auto operator+(const int512 &a, const int512 &b) noexcept -> int512 {
    return _unsigned_add(a, b);
  }

  friend auto operator-(const int512 &a, const int512 &b) noexcept -> int512 {
    return _unsigned_sub(a, b);
  }

  friend auto operator-(const int512 &v) noexcept -> int512 {
    auto t = v;
    t._negate_in_place();
    return t;
  }

  friend auto operator*(const int512 &a, const int512 &b) noexcept -> int512 {
    return _unsigned_mul(a, b);
  }

  [[nodiscard]] friend auto divmod(const int512 &a, const int512 &b) noexcept
      -> std::pair<int512, int512> {
    const auto neg_q = a.is_negative() != b.is_negative();
    const auto neg_r = a.is_negative();
    auto [q, r] = _unsigned_divmod(_abs(a), _abs(b));
    if (neg_q)
      q._negate_in_place();
    if (neg_r && !r.is_zero())
      r._negate_in_place();
    return {q, r};
  }

  friend auto operator/(const int512 &a, const int512 &b) noexcept -> int512 {
    return divmod(a, b).first;
  }

  friend auto operator%(const int512 &a, const int512 &b) noexcept -> int512 {
    return divmod(a, b).second;
  }

  friend constexpr auto operator~(const int512 &v) noexcept -> int512 {
    return int512(~v._limb[0], ~v._limb[1], ~v._limb[2], ~v._limb[3]);
  }

  friend constexpr auto operator&(const int512 &a, const int512 &b) noexcept
      -> int512 {
    return int512(a._limb[0] & b._limb[0], a._limb[1] & b._limb[1],
                  a._limb[2] & b._limb[2], a._limb[3] & b._limb[3]);
  }

  friend constexpr auto operator|(const int512 &a, const int512 &b) noexcept
      -> int512 {
    return int512(a._limb[0] | b._limb[0], a._limb[1] | b._limb[1],
                  a._limb[2] | b._limb[2], a._limb[3] | b._limb[3]);
  }

  friend constexpr auto operator^(const int512 &a, const int512 &b) noexcept
      -> int512 {
    return int512(a._limb[0] ^ b._limb[0], a._limb[1] ^ b._limb[1],
                  a._limb[2] ^ b._limb[2], a._limb[3] ^ b._limb[3]);
  }

  friend auto operator<<(const int512 &a, unsigned s) noexcept -> int512 {
    if (s >= 512)
      return int512(0);
    if (s == 0)
      return a;
    const std::size_t words = s / 128;
    const unsigned bits = s % 128;
    limb_type out[4] = {limb_type(0), limb_type(0), limb_type(0),
                        limb_type(0)};
    for (std::size_t k = 4; k-- > words;) {
      const std::size_t source = k - words;
      limb_type v = a._limb[source] << bits;
      if (bits != 0 && source > 0)
        v |= a._limb[source - 1] >> (128 - bits);
      out[k] = v;
    }
    return _of_limbs(out[0], out[1], out[2], out[3]);
  }

  friend auto operator>>(const int512 &a, unsigned s) noexcept -> int512 {
    if (s >= 512)
      return a.is_negative() ? int512(-1) : int512(0);
    if (s == 0)
      return a;
    const auto fill = a.is_negative() ? ~limb_type(0) : limb_type(0);
    const std::size_t words = s / 128;
    const unsigned bits = s % 128;
    limb_type out[4] = {fill, fill, fill, fill};
    for (std::size_t k = 0; k + words < 4; ++k) {
      const std::size_t source = k + words;
      limb_type v =
          source + 1 < 4
              ? a._limb[source] >> bits
              : static_cast<limb_type>(
                    static_cast<tf::exact::int128>(a._limb[3]) >> bits);
      if (bits != 0 && source + 1 < 4)
        v |= a._limb[source + 1] << (128 - bits);
      out[k] = v;
    }
    return _of_limbs(out[0], out[1], out[2], out[3]);
  }

  auto operator+=(const int512 &o) noexcept -> int512 & {
    return *this = *this + o;
  }
  auto operator-=(const int512 &o) noexcept -> int512 & {
    return *this = *this - o;
  }
  auto operator*=(const int512 &o) noexcept -> int512 & {
    return *this = *this * o;
  }
  auto operator/=(const int512 &o) noexcept -> int512 & {
    return *this = *this / o;
  }
  auto operator%=(const int512 &o) noexcept -> int512 & {
    return *this = *this % o;
  }
  auto operator&=(const int512 &o) noexcept -> int512 & {
    return *this = *this & o;
  }
  auto operator|=(const int512 &o) noexcept -> int512 & {
    return *this = *this | o;
  }
  auto operator^=(const int512 &o) noexcept -> int512 & {
    return *this = *this ^ o;
  }
  auto operator<<=(unsigned s) noexcept -> int512 & {
    return *this = *this << s;
  }
  auto operator>>=(unsigned s) noexcept -> int512 & {
    return *this = *this >> s;
  }

  auto operator++() noexcept -> int512 & { return *this += int512(1); }
  auto operator++(int) noexcept -> int512 {
    auto t = *this;
    ++(*this);
    return t;
  }
  auto operator--() noexcept -> int512 & { return *this -= int512(1); }
  auto operator--(int) noexcept -> int512 {
    auto t = *this;
    --(*this);
    return t;
  }
};

} // namespace tf::exact

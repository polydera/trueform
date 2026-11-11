/**
 * @file uniform_draw.hpp
 * @brief Random draws from an engine's own words, the same on every
 * standard library
 *
 * The engines are standardized and the distributions are not, so a test
 * seeds a std engine and draws through these.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace tf::test {

/// Sixty-four bits of the engine's output: two words of a 32-bit engine,
/// the first drawn the high one.
template <typename Engine> auto uniform_word(Engine &rng) -> std::uint64_t {
  if constexpr (Engine::max() <= 0xffffffffull) {
    const auto high = std::uint64_t(rng());
    const auto low = std::uint64_t(rng());
    return (high << 32) | low;
  } else {
    return std::uint64_t(rng());
  }
}

/// An integer in [lo, hi].
template <typename Engine, typename Int>
auto uniform_int(Engine &rng, Int lo, Int hi) -> Int {
  static_assert(std::is_integral_v<Int>);
  const auto span = std::uint64_t(std::int64_t(hi) - std::int64_t(lo)) + 1;
  const auto word = uniform_word(rng);
  return Int(std::int64_t(lo) + std::int64_t(span ? word % span : word));
}

/// A real in [lo, hi).
template <typename Engine, typename Real>
auto uniform_real(Engine &rng, Real lo, Real hi) -> Real {
  static_assert(std::is_floating_point_v<Real>);
  const auto unit = double(uniform_word(rng) >> 11) * 0x1.0p-53;
  return Real(lo + (hi - lo) * Real(unit));
}

/// True with probability `p`.
template <typename Engine> auto uniform_coin(Engine &rng, double p) -> bool {
  return double(uniform_word(rng) >> 11) * 0x1.0p-53 < p;
}

/// An approximately normal real of `mean` and `sigma`: the sum of twelve
/// unit draws less six, in plain arithmetic so no libm enters.
template <typename Engine, typename Real>
auto normal_real(Engine &rng, Real mean, Real sigma) -> Real {
  static_assert(std::is_floating_point_v<Real>);
  double sum = -6.0;
  for (int k = 0; k < 12; ++k)
    sum += double(uniform_word(rng) >> 11) * 0x1.0p-53;
  return Real(mean + sigma * Real(sum));
}

/// The range permuted by Fisher-Yates over `uniform_int`.
template <typename Engine, typename Iterator>
auto shuffle(Iterator first, Iterator last, Engine &rng) -> void {
  const auto n = std::size_t(std::distance(first, last));
  for (std::size_t i = n; i > 1; --i) {
    const auto j = uniform_int(rng, std::size_t(0), i - 1);
    using std::swap;
    swap(first[std::ptrdiff_t(i - 1)], first[std::ptrdiff_t(j)]);
  }
}

} // namespace tf::test

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

#include "../checked.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

namespace tf {

/// @ingroup core_algorithms
/// @brief Simplified parallel reduction using a single binary operator.
///
/// Reduces a range to a single value using the provided binary operator.
///
/// @tparam Range The input range type.
/// @tparam F A binary operator: `Val(Val, element_type)`.
/// @tparam Val The accumulator type.
/// @param r The input range to reduce.
/// @param f Binary operator for combining elements.
/// @param initial The initial accumulator value.
/// @return The reduced result.
template <typename Range, typename F, typename Val>
auto reduce(const Range &r, const F &f, Val initial) {
  const std::size_t n = r.size();
  if (n == 0)
    return initial;
  std::size_t grain = std::max<std::size_t>(
      1, n / (std::max<unsigned>(1, std::thread::hardware_concurrency()) * 5));
  auto begin = r.begin();
  return tbb::parallel_reduce(
      tbb::blocked_range<std::size_t>(0, n, grain), initial,
      [&begin, &f](const tbb::blocked_range<std::size_t> &sub, Val acc) {
        for (std::size_t i = sub.begin(); i < sub.end(); ++i)
          acc = f(acc, *(begin + i));
        return acc;
      },
      [&f](Val a, const Val &b) { return f(a, b); });
}

/// @ingroup core_algorithms
/// @brief Simplified parallel reduction with checked execution.
///
/// Falls back to sequential execution for ranges smaller than 1000 elements.
/// This is useful for debugging and verification.
template <typename Range, typename F, typename Val>
auto reduce(const Range &r, const F &f, Val initial, tf::checked_t c) {
  if (std::size_t(r.size()) < c.serial_below) {
    for (const auto &x : r)
      initial = f(initial, x);
    return initial;
  } else
    return reduce(r, f, std::move(initial));
}
} // namespace tf

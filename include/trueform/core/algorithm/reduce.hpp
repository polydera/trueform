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
#include "./block_reduce.hpp"

namespace tf {

/// @ingroup core_algorithms
/// @brief Simplified parallel reduction using a single binary operator.
///
/// Reduces a range to a single value using the provided binary operator.
/// This is a simplified wrapper around @ref tf::blocked_reduce that uses
/// the same operator for both the block-level reduction and aggregation.
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
  tf::blocked_reduce(
      r, initial,
      [&f](const auto &r, auto &init) {
        for (auto e : r)
          init = f(init, e);
      },
      [&f](const auto &x, auto &y) { y = f(y, x); });
  return initial;
}

/// @ingroup core_algorithms
/// @brief Simplified parallel reduction with checked execution.
///
/// Falls back to sequential execution for ranges smaller than 1000 elements.
/// This is useful for debugging and verification.
template <typename Range, typename F, typename Val>
auto reduce(const Range &r, const F &f, Val initial, tf::checked_t) {
  if (r.size() < 1000) {
    for (const auto &x : r)
      initial = f(initial, x);
    return initial;
  } else
    return reduce(r, f, std::move(initial));
}
} // namespace tf

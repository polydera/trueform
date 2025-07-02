/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./parallel_for.hpp"
#include <numeric>

namespace tf {
/// @ingroup algorithms
/// @brief Fills a range with sequentially increasing values in parallel.
///
/// This function is a parallel equivalent of `std::iota`, using @ref parallel_for internally
/// to divide the range into blocks and fill them concurrently. Each element is set to an
/// incrementing value starting from `val`, maintaining sequential consistency.
///
/// @tparam Range The type of the range. Must support random-access iteration.
/// @tparam Val The type of the starting value. Must be compatible with `std::iota`.
/// @param r The range to fill.
/// @param val The starting value for the iota sequence.
///
/// @note The elements in the range are filled with: `val, val + 1, val + 2, ...`
///
/// @see @ref parallel_for
template <typename Range, typename Val> auto parallel_iota(Range &&r, Val val) {
  parallel_for(r, [val, start = r.begin()](auto begin, auto end) {
    std::iota(begin, end, val + (begin - start));
  });
}

} // namespace tf

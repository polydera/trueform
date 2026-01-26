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
#include "./parallel_for.hpp"
#include <numeric>

namespace tf {
/// @ingroup core_algorithms
/// @brief Fill a range with sequentially increasing values in parallel.
///
/// This function is a parallel equivalent of `std::iota`, using @ref
/// parallel_for internally to divide the range into blocks and fill them
/// concurrently. Each element is set to an incrementing value starting from
/// `val`, maintaining sequential consistency.
///
/// @tparam Range The type of the range. Must support random-access iteration.
/// @tparam Val The type of the starting value. Must be compatible with
/// `std::iota`.
/// @param r The range to fill.
/// @param val The starting value for the iota sequence.
///
/// @note The elements in the range are filled with: `val, val + 1, val + 2,
/// ...`
///
/// @see @ref parallel_for
template <typename Range, typename Val> auto parallel_iota(Range &&r, Val val) {
  if (r.size() < 1000)
    std::iota(r.begin(), r.end(), val);
  else
    parallel_for(r, [val, start = r.begin()](auto begin, auto end) {
      using value_t = std::decay_t<decltype(*begin)>;
      std::iota(begin, end, static_cast<value_t>(val + (begin - start)));
    });
}

} // namespace tf

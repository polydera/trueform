/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "tbb/parallel_for.h"

namespace tf {
template <typename Iterator, typename Func>
auto parallel_for(Iterator first, Iterator last, Func &&f) -> void {
  tbb::parallel_for(
      tbb::blocked_range<Iterator>(first, last),
      [f = static_cast<Func &&>(f)](const tbb::blocked_range<Iterator> &range) {
        f(range.begin(), range.end());
      });
}
/// @ingroup algorithms
/// @brief Executes a parallel for loop over a container-like range.
///
///
/// @tparam Range Type of the container or range (must provide `.begin()` and `.end()`).
/// @tparam Func Callable type; must accept `(Iterator, Iterator)` arguments.
/// @param r The container or range to iterate over.
/// @param f Function to apply to each subrange in parallel.
template <typename Range, typename Func>
auto parallel_for(Range &&r, Func &&f) -> void {
  return parallel_for(r.begin(), r.end(), static_cast<Func &&>(f));
}
} // namespace tf

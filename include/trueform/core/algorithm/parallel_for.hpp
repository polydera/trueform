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
/// @ingroup core_algorithms
/// @brief Executes a parallel for loop over a container-like range.
///
///
/// @tparam Range Type of the container or range (must provide `.begin()` and
/// `.end()`).
/// @tparam Func Callable type; must accept `(Iterator, Iterator)` arguments.
/// @param r The container or range to iterate over.
/// @param f Function to apply to each subrange in parallel.
template <typename Range, typename Func>
auto parallel_for(Range &&r, Func &&f) -> void {
  return parallel_for(r.begin(), r.end(), static_cast<Func &&>(f));
}

template <typename Range, typename Func>
auto parallel_for(Range &&r, Func &&f, tf::checked_t) -> void {
  if (r.size() < 1000)
    f(r.begin(), r.end());
  else
    parallel_for(r.begin(), r.end(), static_cast<Func &&>(f));
}
} // namespace tf

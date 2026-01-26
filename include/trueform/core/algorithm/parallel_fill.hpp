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

namespace tf {

/// @ingroup core_algorithms
/// @brief Fill a range with a value in parallel.
///
/// Falls back to sequential fill for ranges smaller than 1000 elements.
template <typename Range, typename T>
auto parallel_fill(Range &&r, const T &val) -> void {
  if (r.size() < 1000)
    std::fill(r.begin(), r.end(), val);
  else
    tf::parallel_for(r,
                     [&](auto begin, auto end) { std::fill(begin, end, val); });
}
} // namespace tf

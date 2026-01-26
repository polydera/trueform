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
#include <algorithm>

namespace tf {

/// @ingroup core_algorithms
/// @brief Replace all occurrences of a value in parallel.
///
/// Falls back to sequential replace for ranges smaller than 1000 elements.
template <typename Range, typename T>
auto parallel_replace(Range &&r, const T &val, const T &new_val) -> void {
  if (r.size() < 1000)
    std::replace(r.begin(), r.end(), val, new_val);
  else
    tf::parallel_for(r, [&](auto begin, auto end) {
      std::replace(begin, end, val, new_val);
    });
}
} // namespace tf

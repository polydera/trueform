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
#include <algorithm>

namespace tf {

/// @ingroup core_algorithms
/// @brief Return the smaller of two values.
///
/// @tparam T The value type.
/// @param t0 First value.
/// @param t1 Second value.
/// @return Reference to the smaller value.
template <typename T> auto min(const T &t0, const T &t1) -> const T & {
  using std::min;
  return min(t0, t1);
}
} // namespace tf

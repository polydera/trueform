/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <algorithm>

namespace tf {

/// @ingroup core_algorithms
/// @brief Return the larger of two values.
///
/// @tparam T The value type.
/// @param t0 First value.
/// @param t1 Second value.
/// @return Reference to the larger value.
template <typename T> auto max(const T &t0, const T &t1) -> const T & {
  using std::max;
  return max(t0, t1);
}
} // namespace tf

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <algorithm>

namespace tf {
template <typename T> auto max(const T &t0, const T &t1) -> const T & {
  using std::max;
  return max(t0, t1);
}
} // namespace tf

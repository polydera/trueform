/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <algorithm>

namespace tf {
template <typename T> auto min(const T &t0, const T &t1) -> const T & {
  using std::min;
  return min(t0, t1);
}
} // namespace tf

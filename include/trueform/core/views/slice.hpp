/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../range.hpp"

namespace tf {
template <typename Range>
auto slice(Range &&range, std::size_t _from, std::size_t _to) {
  return tf::make_range(range.begin() + _from, range.begin() + _to);
}
} // namespace tf

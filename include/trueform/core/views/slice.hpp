/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
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

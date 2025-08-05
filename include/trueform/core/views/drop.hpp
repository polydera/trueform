/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../range.hpp"

namespace tf {
template <typename Range> auto drop(Range &&range, std::size_t n) {
  return tf::make_range(range.begin() + n, range.end());
}
} // namespace tf

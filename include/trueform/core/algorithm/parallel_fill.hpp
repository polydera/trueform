/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./parallel_for.hpp"

namespace tf {
template <typename Range, typename T>
auto parallel_fill(Range &&r, const T &val) -> void {
  if (r.size() < 1000)
    std::fill(r.begin(), r.end(), val);
  else
    tf::parallel_for(r,
                     [&](auto begin, auto end) { std::fill(begin, end, val); });
}
} // namespace tf

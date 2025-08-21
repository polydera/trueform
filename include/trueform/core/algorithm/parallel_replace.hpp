/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./parallel_for.hpp"
#include <algorithm>

namespace tf {
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

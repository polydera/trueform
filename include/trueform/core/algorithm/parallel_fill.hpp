/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
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

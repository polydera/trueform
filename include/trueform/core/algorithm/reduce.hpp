/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./block_reduce.hpp"

namespace tf {
template <typename Range, typename F, typename Val>
auto reduce(const Range &r, const F &f, Val initial) {
  tf::blocked_reduce(
      r, initial,
      [&f](const auto &r, auto &init) {
        for (auto e : r)
          init = f(init, e);
      },
      [&f](const auto &x, auto &y) { y = f(y, x); });
  return initial;
}
} // namespace tf

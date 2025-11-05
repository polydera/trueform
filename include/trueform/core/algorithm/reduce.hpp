/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../checked.hpp"
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

template <typename Range, typename F, typename Val>
auto reduce(const Range &r, const F &f, Val initial, tf::checked_t) {
  if (r.size() < 1000) {
    for (const auto &x : r)
      initial = f(initial, x);
    return initial;
  } else
    return reduce(r, f, std::move(initial));
}
} // namespace tf

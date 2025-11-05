/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../views/zip.hpp"
#include "./parallel_apply.hpp"
#include <algorithm>

namespace tf {

template <typename Range0, typename Range1>
auto parallel_copy_blocked(const Range0 &input, Range1 &&output) {
  if (input.size() < 1000)
    for (auto &&[in, out] : tf::zip(input, output))
      std::copy(in.begin(), in.end(), out.begin());
  else
    tf::parallel_apply(
        tf::zip(input, output),
        [](auto &&pair) {
          auto &&[in, out] = pair;
          std::copy(in.begin(), in.end(), out.begin());
        },
        tf::checked);
}

template <typename Range0, typename Range1>
auto parallel_copy_blocked_reverse(const Range0 &input, Range1 &&output) {
  if (input.size() < 1000)
    for (auto &&[in, out] : tf::zip(input, output))
      std::reverse_copy(in.begin(), in.end(), out.begin());
  else
    tf::parallel_apply(
        tf::zip(input, output),
        [](auto &&pair) {
          auto &&[in, out] = pair;
          std::reverse_copy(in.begin(), in.end(), out.begin());
        },
        tf::checked);
}

} // namespace tf

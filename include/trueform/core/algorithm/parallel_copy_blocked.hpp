/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#pragma once
#include "../views/zip.hpp"
#include "./parallel_for_each.hpp"
#include <algorithm>

namespace tf {

/// @ingroup core_algorithms
/// @brief Parallel copy of blocked ranges.
///
/// Copies each block from input to corresponding output block.
/// Falls back to sequential for small inputs (<1000 blocks).
///
/// @tparam Range0 The input blocked range type.
/// @tparam Range1 The output blocked range type.
/// @param input Source blocked range.
/// @param output Destination blocked range.
template <typename Range0, typename Range1>
auto parallel_copy_blocked(const Range0 &input, Range1 &&output) {
  if (input.size() < 1000)
    for (auto &&[in, out] : tf::zip(input, output))
      std::copy(in.begin(), in.end(), out.begin());
  else
    tf::parallel_for_each(
        tf::zip(input, output),
        [](auto &&pair) {
          auto &&[in, out] = pair;
          std::copy(in.begin(), in.end(), out.begin());
        },
        tf::checked);
}

/// @ingroup core_algorithms
/// @brief Parallel copy of blocked ranges with reversal.
///
/// Copies each block in reverse order from input to output.
///
/// @tparam Range0 The input blocked range type.
/// @tparam Range1 The output blocked range type.
/// @param input Source blocked range.
/// @param output Destination blocked range.
template <typename Range0, typename Range1>
auto parallel_copy_blocked_reverse(const Range0 &input, Range1 &&output) {
  if (input.size() < 1000)
    for (auto &&[in, out] : tf::zip(input, output))
      std::reverse_copy(in.begin(), in.end(), out.begin());
  else
    tf::parallel_for_each(
        tf::zip(input, output),
        [](auto &&pair) {
          auto &&[in, out] = pair;
          std::reverse_copy(in.begin(), in.end(), out.begin());
        },
        tf::checked);
}

} // namespace tf

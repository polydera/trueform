/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include <cstddef>

namespace tf::csg::graph {

/// @brief Per-tag sheet flags from the declared operand ids. Empty when
///        nothing is declared, which every consumer reads as "no
///        sheets" without touching the mask.
template <typename Iter, std::size_t N, typename Index>
auto make_sheet_mask(tf::range<Iter, N> sheets, Index n_tags)
    -> tf::buffer<char> {
  tf::buffer<char> is_sheet;
  if (sheets.size() == 0)
    return is_sheet;
  is_sheet.allocate(static_cast<std::size_t>(n_tags));
  tf::parallel_fill(is_sheet, char(0));
  for (auto id : sheets)
    is_sheet[static_cast<std::size_t>(id)] = char(1);
  return is_sheet;
}

} // namespace tf::csg::graph

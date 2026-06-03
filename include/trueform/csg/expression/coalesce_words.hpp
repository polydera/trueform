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
#include "./compiled_expr.hpp"
#include <algorithm>
#include <vector>

namespace tf::csg::detail {

/// @ingroup csg
/// @brief Group `word_mask` entries by word index, OR'ing same-word
///        masks. Output is sorted ascending by word index.
inline auto coalesce_words(std::vector<compiled_expr::word_mask> v)
    -> std::vector<compiled_expr::word_mask> {
  std::sort(v.begin(), v.end(),
             [](const auto &a, const auto &b) { return a.idx < b.idx; });
  std::vector<compiled_expr::word_mask> out;
  out.reserve(v.size());
  for (const auto &wm : v) {
    if (!out.empty() && out.back().idx == wm.idx)
      out.back().mask |= wm.mask;
    else
      out.push_back(wm);
  }
  return out;
}

} // namespace tf::csg::detail

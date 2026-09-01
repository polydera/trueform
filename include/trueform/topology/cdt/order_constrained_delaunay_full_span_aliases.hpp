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

#include "../../core/buffer.hpp"
#include "./constrained_delaunay_full_span_alias.hpp"
#include "./size_adaptive.hpp"

#include <array>
#include <cstddef>
#include <tuple>

namespace tf::topology::cdt {

/// Order prepared complete spans and publish only their multiplicity-greater-
/// than-one blocks. Collapsed constraints never enter the carrier; partial
/// overlaps have different endpoint keys and remain recovery facts.
template <typename Index>
auto order_constrained_delaunay_full_span_aliases(
    tf::buffer<constrained_delaunay_full_span_alias<Index>> &spans,
    tf::buffer<std::array<Index, 2>> &alias_blocks) -> void {
  using span_t = constrained_delaunay_full_span_alias<Index>;
  alias_blocks.clear();
  tf::topology::cdt::sort_auto(
      spans.begin(), spans.end(), [](const span_t &a, const span_t &b) {
        return std::tie(a.first, a.second, a.input) <
               std::tie(b.first, b.second, b.input);
      });

  for (std::size_t begin = 0; begin < spans.size();) {
    std::size_t end = begin + 1;
    while (end < spans.size() && spans[end].first == spans[begin].first &&
           spans[end].second == spans[begin].second)
      ++end;
    if (end - begin > 1)
      alias_blocks.push_back(
          {static_cast<Index>(begin), static_cast<Index>(end)});
    begin = end;
  }
}

} // namespace tf::topology::cdt

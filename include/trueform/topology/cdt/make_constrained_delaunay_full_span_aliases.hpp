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
#include "../../core/edges.hpp"
#include "./append_constrained_delaunay_full_span_alias.hpp"
#include "./constrained_delaunay_full_span_alias.hpp"
#include "./order_constrained_delaunay_full_span_aliases.hpp"

#include <array>
#include <cstddef>

namespace tf::topology::cdt {

/// Materialize the one prepared full-span carrier shared by CDT consumers.
template <typename Index, typename EdgesPolicy, typename InputToOutput>
auto make_constrained_delaunay_full_span_aliases(
    const tf::edges<EdgesPolicy> &edges, const InputToOutput &input_to_output,
    tf::buffer<constrained_delaunay_full_span_alias<Index>> &spans,
    tf::buffer<std::array<Index, 2>> &alias_blocks) -> void {
  spans.clear();
  spans.reserve(std::size_t(edges.size()));
  for (Index input = 0; input < static_cast<Index>(edges.size()); ++input) {
    const Index mapped_first =
        input_to_output[std::size_t(edges[input][0])];
    const Index mapped_second =
        input_to_output[std::size_t(edges[input][1])];
    append_constrained_delaunay_full_span_alias(
        mapped_first, mapped_second, input, spans);
  }
  order_constrained_delaunay_full_span_aliases(spans, alias_blocks);
}

} // namespace tf::topology::cdt

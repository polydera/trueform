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

#include <algorithm>

namespace tf::topology::cdt {

template <typename Index>
auto append_constrained_delaunay_full_span_alias(
    Index first, Index second, Index input,
    tf::buffer<constrained_delaunay_full_span_alias<Index>> &spans) -> void {
  if (first == second)
    return;
  spans.push_back({std::min(first, second), std::max(first, second), input,
                   char(first > second)});
}

} // namespace tf::topology::cdt

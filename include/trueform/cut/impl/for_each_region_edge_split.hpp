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

#include "../../core/buffer.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_identity.hpp"

namespace tf::cut {

template <typename Index, typename Split, typename Emit>
auto for_each_region_edge_split(
    Index n_tags, Index tag,
    const tf::intersect::graph::vertex<Index> &a,
    const tf::intersect::graph::vertex<Index> &b,
    const tf::buffer<Split> &splits, const Emit &emit) -> void {
  const auto a_key = tf::cut::detail::region_vertex_key(n_tags, tag, a);
  const auto b_key = tf::cut::detail::region_vertex_key(n_tags, tag, b);
  const auto range = tf::cut::detail::find_region_edge_splits(
      tf::cut::detail::region_edge_key(a_key, b_key), splits);
  const bool forward = a_key < b_key;
  for (Index offset = 0; offset < range.second - range.first; ++offset)
    emit(splits[std::size_t(forward ? range.first + offset
                                    : range.second - 1 - offset)]
             .vertex);
}

} // namespace tf::cut

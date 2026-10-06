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
#include "./arrangement_descriptor.hpp"
#include "./triangle_component_labels.hpp"
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Per form, whether it is a volume none of whose components the
///        open mask names: its crossing count is then a winding number, a
///        function of position alone.
template <typename Index>
auto make_closed_volume_tags(
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const tf::csg::graph::triangle_component_labels<Index> &labels,
    Index n_tags, const tf::buffer<char> &is_sheet_tag) -> tf::buffer<char> {
  tf::buffer<char> closed;
  closed.allocate(static_cast<std::size_t>(n_tags));
  tf::parallel_fill(closed, char(1));
  for (Index t = Index(0); t < Index(is_sheet_tag.size()) && t < n_tags; ++t)
    if (is_sheet_tag[t])
      closed[t] = char(0);
  auto open_mask = labels.open_component_mask();
  for (Index c = Index(0); c < labels.n_components(); ++c) {
    const Index t = desc.tag_of_component[c];
    if (t >= Index(0) && open_mask[c])
      closed[t] = char(0);
  }
  return closed;
}

} // namespace tf::csg::graph

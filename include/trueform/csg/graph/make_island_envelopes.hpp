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
#include "./arrangement_descriptor.hpp"
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Per island, its outer env: the most-negative-volume domain its
///        components bound — the region the island stands in.
template <typename Index, typename VolT>
auto make_island_envelopes(
    const tf::buffer<Index> &island_of_component, Index n_islands,
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const tf::buffer<VolT> &domain_volumes) -> tf::buffer<Index> {
  tf::buffer<Index> outer_env;
  outer_env.allocate(static_cast<std::size_t>(n_islands));
  for (Index i = Index(0); i < n_islands; ++i)
    outer_env[i] = Index(-1);
  const Index n_components = Index(island_of_component.size());
  for (Index c = Index(0); c < n_components; ++c) {
    const Index i = island_of_component[c];
    for (Index s = Index(0); s < Index(2); ++s) {
      const Index d = desc.domain_of_side[2 * c + s];
      if (outer_env[i] == Index(-1) ||
          domain_volumes[d] < domain_volumes[outer_env[i]])
        outer_env[i] = d;
    }
  }
  return outer_env;
}

} // namespace tf::csg::graph

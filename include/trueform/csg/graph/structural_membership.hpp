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
#include "./union_find.hpp"

#include <array>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief The unbounded universe: the domain with the most negative
///        exact signed volume (the seeder's null-seed rule).
///
/// @pre `volumes` is non-empty.
template <typename Volumes>
auto find_universe_domain(const Volumes &volumes) -> std::size_t {
  std::size_t universe = 0;
  for (std::size_t d = 1; d < volumes.size(); ++d)
    if (volumes[d] < volumes[universe])
      universe = d;
  return universe;
}

/// @ingroup csg_graph_internals
/// @brief Fine-domain membership of the structural outer read: false
///        exactly on the universe CLASS.
///
/// The universe is one region but several fine domains for contact-free
/// components (disjoint or nested shells), related only through the
/// seeder's nesting merges — without uniting them first, a disjoint
/// component's outside reads as interior and its wall drops from the
/// shell.
template <typename Index, typename Volumes>
auto compute_structural_membership(
    const Volumes &volumes,
    const tf::buffer<std::array<Index, 2>> &nesting_merges)
    -> tf::buffer<bool> {
  tf::buffer<bool> membership;
  if (volumes.size() == 0)
    return membership;
  tf::csg::graph::union_find<Index> uf;
  uf.reset(volumes.size());
  for (const auto &m : nesting_merges)
    uf.unite(m[0], m[1]);
  const Index universe_class = uf.find(Index(find_universe_domain(volumes)));
  membership.allocate(volumes.size());
  for (std::size_t d = 0; d < membership.size(); ++d)
    membership[d] = uf.find(Index(d)) != universe_class;
  return membership;
}

} // namespace tf::csg::graph

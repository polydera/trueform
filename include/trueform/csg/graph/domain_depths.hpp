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
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Per-domain signed layer depths, one column per form.
///
/// Column `t` of domain `d` counts form `t`'s walls between the universe
/// and `d`, signed by the side the crossing enters: a wall's side 1 is the
/// side its faces are wound away from (the side
/// @ref tf::csg::graph::compute_arrangement_domain_volumes gives `+C`), so
/// crossing `0 -> 1` adds a layer and `1 -> 0` removes one. The universe
/// stands at zero, and `depth != 0` is what
/// @ref tf::csg::graph::domain_inclusions publishes as inside.
///
/// A declared sheet encloses nothing and has no depth: its column is never
/// written, its inclusion staying the bit its winding seed and the flood's
/// XOR state. The rows are transient — the flood publishes them and the
/// carrier dies.
template <typename Index> struct domain_depths {
  tf::buffer<Index> depths;
  std::size_t columns_per_domain = 0;

  /// @brief Form `tag`'s depth in domain `d`.
  auto at(std::size_t d, std::size_t tag) -> Index & {
    return depths[d * columns_per_domain + tag];
  }

  /// @overload
  auto at(std::size_t d, std::size_t tag) const -> const Index & {
    return depths[d * columns_per_domain + tag];
  }
};

/// @ingroup csg_graph_internals
/// @brief The depth rows of an N-form arrangement, in the state every
///        depth is counted from: the universe's, outside everything.
///
/// @param n_tags    The number of forms in the arrangement.
/// @param n_domains The arrangement descriptor's domain count.
template <typename Index>
auto make_domain_depths(Index n_tags, Index n_domains) -> domain_depths<Index> {
  domain_depths<Index> out;
  out.columns_per_domain = static_cast<std::size_t>(n_tags);
  out.depths.allocate(static_cast<std::size_t>(n_domains) *
                      out.columns_per_domain);
  tf::parallel_fill(out.depths, Index(0));
  return out;
}

} // namespace tf::csg::graph

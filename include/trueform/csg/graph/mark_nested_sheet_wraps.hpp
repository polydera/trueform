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
#include "./sheet_sides.hpp"
#include "./union_find.hpp"
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Mark the free rims' wraps and carry every wrap across the nesting
///        merges — the nested witness of @ref tf::csg::graph::sheet_sides.
///
/// `rim_wraps` are the `(domain, tag)` landings of the bundles holding a
/// sheet's free rim (@ref tf::csg::graph::seed_inclusion_bits).
template <typename Index>
auto mark_nested_sheet_wraps(
    tf::csg::graph::sheet_sides<Index> &sides,
    const tf::buffer<std::array<Index, 2>> &rim_wraps,
    const tf::buffer<std::array<Index, 2>> &nesting_merges) -> void {
  using sides_t = tf::csg::graph::sheet_sides<Index>;
  const Index n_sheets = sides.n_sheets();
  if (n_sheets == Index(0))
    return;
  for (const auto &rim : rim_wraps)
    sides.state(sides.sheet_of_tag[rim[1]], rim[0]) = sides_t::wraps;
  if (nesting_merges.size() == 0)
    return;

  tf::csg::graph::union_find<Index> classes;
  classes.reset(static_cast<std::size_t>(sides.n_domains));
  for (const auto &m : nesting_merges)
    classes.unite(m[0], m[1]);
  // only the merged domains move, so the classes are read off them alone
  for (Index k = Index(0); k < n_sheets; ++k) {
    for (const auto &m : nesting_merges)
      for (const Index d : m)
        if (sides.state(k, d) == sides_t::wraps)
          sides.state(k, classes.find(d)) = sides_t::wraps;
    for (const auto &m : nesting_merges)
      for (const Index d : m)
        if (sides.state(k, classes.find(d)) == sides_t::wraps)
          sides.state(k, d) = sides_t::wraps;
  }
}

} // namespace tf::csg::graph

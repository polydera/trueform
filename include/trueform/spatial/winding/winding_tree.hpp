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
#include <cassert>

namespace tf::spatial {

/// The moments-capable tree the winding traversal walks: the tree's
/// nodes and ids beside the moment rows that mirror the nodes. Sealed
/// here so a tree cannot meet another tree's moments — every query
/// assembles it through @ref make_winding_tree and nowhere else.
template <typename Nodes, typename Ids, typename Moments> struct winding_tree {
  Nodes nodes;
  Ids ids;
  Moments moments;
};

template <typename TreeLike, typename Moments>
auto make_winding_tree(const TreeLike &tree, const Moments &moments) {
  assert(moments.size() == tree.nodes().size() &&
         "the winding moments were built over a different tree");
  return winding_tree<decltype(tree.nodes()), decltype(tree.ids()), Moments>{
      tree.nodes(), tree.ids(), moments};
}

} // namespace tf::spatial

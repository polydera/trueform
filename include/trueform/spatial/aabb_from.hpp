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
#include "../core/aabb_from.hpp"
#include "./mod_tree_like.hpp"
#include "./tree_like.hpp"

namespace tf {

/// @ingroup spatial_queries
/// @brief Construct an AABB from a tree's root bounding volume.
///
/// Returns the AABB enclosing the root BV (aabb, obb, or obbrss).
/// Returns an empty AABB if the tree has no nodes.
template <typename Policy>
auto aabb_from(const tree_like<Policy> &tree) {
  using T = typename Policy::coordinate_type;
  constexpr auto Dims = Policy::coordinate_dims::value;
  if (!tree.nodes().size())
    return make_empty_aabb<T, Dims>();
  return aabb_from(tree.bv());
}

/// @ingroup spatial_queries
/// @brief Construct an AABB from a mod_tree's combined bounding volume.
///
/// Returns the union of the main tree and delta tree AABBs.
template <typename Policy>
auto aabb_from(const mod_tree_like<Policy> &tree) {
  return aabb_union(aabb_from(tree.main_tree()), aabb_from(tree.delta_tree()));
}

} // namespace tf

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
#include "../../core/none.hpp"
#include "../../core/polygons.hpp"
#include "../../spatial/aabb_tree.hpp"
#include "../../spatial/policy/tree.hpp"
#include "../../topology/face_membership.hpp"
#include "../../topology/manifold_edge_link.hpp"
#include "../../topology/policy/manifold_edge_link.hpp"
#include "tbb/parallel_invoke.h"
#include <tuple>

namespace tf::arrangement::dispatch {

/// Build missing boolean structures for polygons.
/// Returns tf::none if all structures present, or tuple of built structures.
template <typename Policy>
auto make_missing_structures(const tf::polygons<Policy> &polygons) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  constexpr auto FaceSize = tf::static_size_v<decltype(polygons.faces()[0])>;
  using CoordType = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;

  if constexpr (tf::has_tree_policy<Policy> &&
                tf::has_manifold_edge_link_policy<Policy>) {
    return tf::none;
  } else if constexpr (!tf::has_tree_policy<Policy> &&
                       !tf::has_manifold_edge_link_policy<Policy>) {
    tf::aabb_tree<Index, CoordType, Dims> tree;
    tf::face_membership<Index> fm;
    tf::manifold_edge_link<Index, FaceSize> mel;
    tbb::parallel_invoke(
        [&] { tree.build(polygons, tf::config_tree(4, 12)); },
        [&] {
          fm.build(polygons);
          mel.build(polygons.faces(), fm);
        });
    return std::make_tuple(std::move(fm), std::move(mel), std::move(tree));
  } else if constexpr (!tf::has_tree_policy<Policy>) {
    tf::aabb_tree<Index, CoordType, Dims> tree;
    tree.build(polygons, tf::config_tree(4, 12));
    return std::make_tuple(std::move(tree));
  } else {
    // Only MEL missing
    tf::face_membership<Index> fm;
    tf::manifold_edge_link<Index, FaceSize> mel;
    fm.build(polygons);
    mel.build(polygons.faces(), fm);
    return std::make_tuple(std::move(fm), std::move(mel));
  }
}

} // namespace tf::arrangement::dispatch

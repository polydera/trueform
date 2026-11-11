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
#include "../../topology/policy/face_membership.hpp"
#include "../../topology/policy/manifold_edge_link.hpp"
#include "tbb/parallel_invoke.h"
#include <tuple>

namespace tf::arrangement::dispatch {

/// Build the structures an arrangement reads that `polygons` does not carry:
/// `tf::none` when it carries its tree, face membership and manifold edge
/// link, otherwise the tuple of what was built — the face membership and the
/// link, each where missing (a missing link built from the membership the
/// form carries or the one built here), then the tree, built beside them.
template <typename Policy>
auto make_missing_structures(const tf::polygons<Policy> &polygons) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  constexpr auto FaceSize = tf::static_size_v<decltype(polygons.faces()[0])>;
  using CoordType = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  constexpr bool has_tree = tf::has_tree_policy<Policy>;
  constexpr bool has_membership = tf::has_face_membership_policy<Policy>;
  constexpr bool has_link = tf::has_manifold_edge_link_policy<Policy>;

  if constexpr (has_tree && has_membership && has_link) {
    return tf::none;
  } else {
    tf::aabb_tree<Index, CoordType, Dims> tree;
    tf::face_membership<Index> fm;
    tf::manifold_edge_link<Index, FaceSize> mel;
    const auto topology = [&] {
      if constexpr (!tf::has_face_membership_policy<Policy>)
        fm.build(polygons);
      if constexpr (!tf::has_manifold_edge_link_policy<Policy>) {
        if constexpr (tf::has_face_membership_policy<Policy>)
          mel.build(polygons.faces(), polygons.face_membership());
        else
          mel.build(polygons.faces(), fm);
      }
    };
    if constexpr (!tf::has_tree_policy<Policy> &&
                  (!tf::has_face_membership_policy<Policy> ||
                   !tf::has_manifold_edge_link_policy<Policy>))
      tbb::parallel_invoke(
          [&] { tree.build(polygons, tf::config_tree(4, 12)); }, topology);
    else if constexpr (!tf::has_tree_policy<Policy>)
      tree.build(polygons, tf::config_tree(4, 12));
    else
      topology();
    const auto membership_part = [&] {
      if constexpr (!tf::has_face_membership_policy<Policy>)
        return std::make_tuple(std::move(fm));
      else
        return std::tuple<>{};
    };
    const auto link_part = [&] {
      if constexpr (!tf::has_manifold_edge_link_policy<Policy>)
        return std::make_tuple(std::move(mel));
      else
        return std::tuple<>{};
    };
    const auto tree_part = [&] {
      if constexpr (!tf::has_tree_policy<Policy>)
        return std::make_tuple(std::move(tree));
      else
        return std::tuple<>{};
    };
    return std::tuple_cat(membership_part(), link_part(), tree_part());
  }
}

} // namespace tf::arrangement::dispatch

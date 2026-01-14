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

namespace tf::cut::impl {

/// Tag polygons with all structures from a tuple
template <typename Policy, typename... Ts>
auto tag_with_structures(const tf::polygons<Policy> &p, std::tuple<Ts...> &t) {
  return std::apply([&p](auto &...structs) { return (p | ... | tf::tag(structs)); },
                    t);
}

/// Build missing boolean structures for polygons.
/// Returns tf::none if all structures present, or tuple of built structures.
template <typename Policy>
auto make_missing_boolean_structures(const tf::polygons<Policy> &polygons) {
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
        [&] { tree.build(polygons, tf::config_tree(4, 4)); },
        [&] {
          fm.build(polygons);
          mel.build(polygons.faces(), fm);
        });
    return std::make_tuple(std::move(fm), std::move(mel), std::move(tree));
  } else if constexpr (!tf::has_tree_policy<Policy>) {
    tf::aabb_tree<Index, CoordType, Dims> tree;
    tree.build(polygons, tf::config_tree(4, 4));
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

template <typename Policy0, typename Policy1, typename F>
auto boolean_dispatch(const tf::polygons<Policy0> &_polygons0,
                      const tf::polygons<Policy1> &_polygons1, F &&f) {
  using S0 = decltype(make_missing_boolean_structures(_polygons0));
  using S1 = decltype(make_missing_boolean_structures(_polygons1));

  if constexpr (std::is_same_v<S0, tf::none_t> &&
                std::is_same_v<S1, tf::none_t>) {
    // Both have all required structures
    return f(_polygons0, _polygons1);
  } else if constexpr (!std::is_same_v<S0, tf::none_t> &&
                       !std::is_same_v<S1, tf::none_t>) {
    // Both need building - do in parallel
    S0 s0;
    S1 s1;
    tbb::parallel_invoke(
        [&] { s0 = make_missing_boolean_structures(_polygons0); },
        [&] { s1 = make_missing_boolean_structures(_polygons1); });
    return f(tag_with_structures(_polygons0, s0),
             tag_with_structures(_polygons1, s1));
  } else if constexpr (!std::is_same_v<S0, tf::none_t>) {
    // Only polygons0 needs building
    auto s0 = make_missing_boolean_structures(_polygons0);
    return f(tag_with_structures(_polygons0, s0), _polygons1);
  } else {
    // Only polygons1 needs building
    auto s1 = make_missing_boolean_structures(_polygons1);
    return f(_polygons0, tag_with_structures(_polygons1, s1));
  }
}

} // namespace tf::cut::impl

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/polygons.hpp"
#include "../../spatial/aabb_tree.hpp"
#include "../../spatial/policy/tree.hpp"
#include "../../topology/face_membership.hpp"
#include "../../topology/manifold_edge_link.hpp"
#include "../../topology/policy/face_membership.hpp"
#include "../../topology/policy/manifold_edge_link.hpp"
#include "tbb/parallel_invoke.h"

namespace tf::cut::impl {

template <typename Policy0, typename Policy1, typename F>
auto boolean_dispatch(const tf::polygons<Policy0> &_polygons0,
                      const tf::polygons<Policy1> &_polygons1, F &&f) {
  // Check polygons0 structures
  if constexpr (!tf::has_tree_policy<Policy0> &&
                !tf::has_manifold_edge_link_policy<Policy0>) {
    using Index = std::decay_t<decltype(_polygons0.faces()[0][0])>;
    tf::aabb_tree<Index, tf::coordinate_type<Policy0>,
                  tf::coordinate_dims_v<Policy0>>
        tree;
    tf::face_membership<Index> fm;
    tf::manifold_edge_link<Index,
                           tf::static_size_v<decltype(_polygons0.faces()[0])>>
        mel;
    tbb::parallel_invoke([&] { tree.build(_polygons0, tf::config_tree(4, 4)); },
                         [&] {
                           fm.build(_polygons0);
                           mel.build(_polygons0.faces(), fm);
                         });
    return boolean_dispatch(_polygons0 | tf::tag(fm) | tf::tag(mel) |
                                tf::tag(tree),
                            _polygons1, std::forward<F>(f));
  } else if constexpr (!tf::has_tree_policy<Policy0>) {
    using Index = std::decay_t<decltype(_polygons0.faces()[0][0])>;
    tf::aabb_tree<Index, tf::coordinate_type<Policy0>,
                  tf::coordinate_dims_v<Policy0>>
        tree;
    tree.build(_polygons0, tf::config_tree(4, 4));
    return boolean_dispatch(_polygons0 | tf::tag(tree), _polygons1,
                            std::forward<F>(f));
  } else if constexpr (!tf::has_manifold_edge_link_policy<Policy0>) {
    using Index = std::decay_t<decltype(_polygons0.faces()[0][0])>;
    tf::face_membership<Index> fm;
    tf::manifold_edge_link<Index,
                           tf::static_size_v<decltype(_polygons0.faces()[0])>>
        mel;
    fm.build(_polygons0);
    mel.build(_polygons0.faces(), fm);
    return boolean_dispatch(_polygons0 | tf::tag(fm) | tf::tag(mel), _polygons1,
                            std::forward<F>(f));
  }
  // Check polygons1 structures
  else if constexpr (!tf::has_tree_policy<Policy1> &&
                     !tf::has_manifold_edge_link_policy<Policy1>) {
    using Index = std::decay_t<decltype(_polygons1.faces()[0][0])>;
    tf::aabb_tree<Index, tf::coordinate_type<Policy1>,
                  tf::coordinate_dims_v<Policy1>>
        tree;
    tf::face_membership<Index> fm;
    tf::manifold_edge_link<Index,
                           tf::static_size_v<decltype(_polygons1.faces()[0])>>
        mel;
    tbb::parallel_invoke([&] { tree.build(_polygons1, tf::config_tree(4, 4)); },
                         [&] {
                           fm.build(_polygons1);
                           mel.build(_polygons1.faces(), fm);
                         });
    return boolean_dispatch(
        _polygons0, _polygons1 | tf::tag(fm) | tf::tag(mel) | tf::tag(tree),
        std::forward<F>(f));
  } else if constexpr (!tf::has_tree_policy<Policy1>) {
    using Index = std::decay_t<decltype(_polygons1.faces()[0][0])>;
    tf::aabb_tree<Index, tf::coordinate_type<Policy1>,
                  tf::coordinate_dims_v<Policy1>>
        tree;
    tree.build(_polygons1, tf::config_tree(4, 4));
    return boolean_dispatch(_polygons0, _polygons1 | tf::tag(tree),
                            std::forward<F>(f));
  } else if constexpr (!tf::has_manifold_edge_link_policy<Policy1>) {
    using Index = std::decay_t<decltype(_polygons1.faces()[0][0])>;
    tf::face_membership<Index> fm;
    tf::manifold_edge_link<Index,
                           tf::static_size_v<decltype(_polygons1.faces()[0])>>
        mel;
    fm.build(_polygons1);
    mel.build(_polygons1.faces(), fm);
    return boolean_dispatch(_polygons0, _polygons1 | tf::tag(fm) | tf::tag(mel),
                            std::forward<F>(f));
  } else {
    // Both polygons have all required structures
    return f(_polygons0, _polygons1);
  }
}

} // namespace tf::cut::impl

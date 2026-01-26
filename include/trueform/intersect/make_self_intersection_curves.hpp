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
#include "../core/curves_buffer.hpp"
#include "../spatial/aabb_tree.hpp"
#include "../spatial/policy/tree.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "../topology/face_membership.hpp"
#include "../topology/manifold_edge_link.hpp"
#include "../topology/policy/face_membership.hpp"
#include "../topology/policy/manifold_edge_link.hpp"
#include "./intersections_within_polygons.hpp"
#include "./make_intersection_edges.hpp"
#include "tbb/parallel_invoke.h"

namespace tf {

/// @ingroup intersect_curves
/// @brief Extract curves where a mesh intersects itself.
///
/// Finds all locations where a mesh's faces intersect each other
/// (excluding adjacent faces) and returns the result as connected curves.
/// Accepts plain @ref tf::polygons or forms with precomputed tree policy
/// (@ref tf::tree or @ref tf::mod_tree) and topology policies
/// (@ref tf::face_membership and @ref tf::manifold_edge_link).
///
/// @tparam Policy The policy type for the mesh.
/// @param _polygons The input @ref tf::polygons (or tagged form).
/// @return A @ref tf::curves_buffer containing connected self-intersection curves.
///
/// @see tf::intersections_within_polygons for low-level access.
template <typename Policy>
auto make_self_intersection_curves(const tf::polygons<Policy> &_polygons) {
  if constexpr (!tf::has_tree_policy<Policy> &&
                !tf::has_manifold_edge_link_policy<Policy>) {
    using Index = std::decay_t<decltype(_polygons.faces()[0][0])>;
    tf::aabb_tree<Index, tf::coordinate_type<Policy>,
                  tf::coordinate_dims_v<Policy>>
        tree;
    tf::face_membership<Index> fm;
    tf::manifold_edge_link<Index,
                           tf::static_size_v<decltype(_polygons.faces()[0])>>
        mel;
    tbb::parallel_invoke([&] { tree.build(_polygons, tf::config_tree(4, 4)); },
                         [&] {
                           fm.build(_polygons);
                           mel.build(_polygons.faces(), fm);
                         });
    return make_self_intersection_curves(_polygons | tf::tag(fm) |
                                         tf::tag(mel) | tf::tag(tree));
  } else if constexpr (!tf::has_tree_policy<Policy>) {
    using Index = std::decay_t<decltype(_polygons.faces()[0][0])>;
    tf::aabb_tree<Index, tf::coordinate_type<Policy>,
                  tf::coordinate_dims_v<Policy>>
        tree;
    tree.build(_polygons, tf::config_tree(4, 4));
    return make_self_intersection_curves(_polygons | tf::tag(tree));
  } else if constexpr (!tf::has_manifold_edge_link_policy<Policy>) {
    using Index = std::decay_t<decltype(_polygons.faces()[0][0])>;
    tf::face_membership<Index> fm;
    tf::manifold_edge_link<Index,
                           tf::static_size_v<decltype(_polygons.faces()[0])>>
        mel;
    fm.build(_polygons);
    mel.build(_polygons.faces(), fm);
    return make_self_intersection_curves(_polygons | tf::tag(fm) |
                                         tf::tag(mel));
  } else {
    using Index = typename Policy::index_type;
    tf::intersections_within_polygons<Index, double,
                                      tf::coordinate_dims_v<Policy>>
        iwp;
    iwp.build(_polygons);
    auto ie = tf::make_intersection_edges(iwp);
    auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
    tf::curves_buffer<Index, tf::coordinate_type<Policy>,
                      tf::coordinate_dims_v<Policy>>
        cb;
    cb.paths_buffer() = std::move(paths);
    cb.points_buffer().allocate(iwp.intersection_points().size());
    tf::parallel_copy(iwp.intersection_points(), cb.points());
    return cb;
  }
}
} // namespace tf

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
#include "../cut/impl/dispatch.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./intersections_between_polygons.hpp"
#include "./make_intersection_edges.hpp"

namespace tf {

/// @ingroup intersect_curves
/// @brief Extract intersection curves where two meshes intersect.
///
/// Computes the geometric intersection between two polygon meshes and
/// returns the result as connected curves. Accepts plain @ref tf::polygons
/// or forms with precomputed tree policy (@ref tf::tree or @ref tf::mod_tree)
/// and topology policies (@ref tf::face_membership and @ref tf::manifold_edge_link).
///
/// @tparam Policy0 The policy type for the first mesh.
/// @tparam Policy1 The policy type for the second mesh.
/// @param _polygons0 The first mesh @ref tf::polygons (or tagged form).
/// @param _polygons1 The second mesh @ref tf::polygons (or tagged form).
/// @return A @ref tf::curves_buffer containing connected intersection curves.
///
/// @see tf::intersections_between_polygons for low-level access.
template <typename Policy0, typename Policy1>
auto make_intersection_curves(const tf::polygons<Policy0> &_polygons0,
                              const tf::polygons<Policy1> &_polygons1) {
  return cut::impl::boolean_dispatch(
      _polygons0, _polygons1, [](const auto &form0, const auto &form1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(form0)>::index_type,
                               typename std::decay_t<decltype(form1)>::index_type>;
        tf::intersections_between_polygons<Index, double,
                                           tf::coordinate_dims_v<std::decay_t<decltype(form0)>>>
            ibp;
        ibp.build(form0, form1);
        auto ie = tf::make_intersection_edges(ibp);
        auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
        tf::curves_buffer<Index,
                          tf::coordinate_type<std::decay_t<decltype(form0)>,
                                              std::decay_t<decltype(form1)>>,
                          tf::coordinate_dims_v<std::decay_t<decltype(form0)>>>
            cb;
        cb.paths_buffer() = std::move(paths);
        cb.points_buffer().allocate(ibp.intersection_points().size());
        tf::parallel_copy(ibp.intersection_points(), cb.points());
        return cb;
      });
}
} // namespace tf

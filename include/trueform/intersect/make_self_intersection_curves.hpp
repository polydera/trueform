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
#include "../topology/connect_edges_to_paths.hpp"
#include "./intersections_within_polygons.hpp"
#include "./make_intersection_edges.hpp"

namespace tf {

/// @ingroup intersect_curves
/// @brief Extract curves where a mesh intersects itself.
///
/// Finds all locations where a mesh's faces intersect each other
/// (excluding adjacent faces) and returns the result as connected curves.
/// Use pipe syntax with @ref tf::tag to add the required tree policy
/// (@ref tf::tree or @ref tf::mod_tree) and topology policies
/// (@ref tf::face_membership and @ref tf::manifold_edge_link).
///
/// @tparam Dims The number of dimensions.
/// @tparam Policy The policy type for the mesh form.
/// @param form The mesh @ref tf::form.
/// @return A @ref tf::curves_buffer containing connected self-intersection curves.
///
/// @see tf::intersections_within_polygons for low-level access.
template <typename Policy>
auto make_self_intersection_curves(const tf::polygons<Policy> &form) {
  using Index = typename Policy::index_type;
  tf::intersections_within_polygons<Index, double,
                                    tf::coordinate_dims_v<Policy>>
      iwp;
  iwp.build(form);
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
} // namespace tf

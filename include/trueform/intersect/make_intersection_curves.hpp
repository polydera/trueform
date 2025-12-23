/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/curves_buffer.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./intersections_between_polygons.hpp"
#include "./make_intersection_edges.hpp"

namespace tf {

/// @ingroup intersect_curves
/// @brief Extract intersection curves where two meshes intersect.
///
/// Computes the geometric intersection between two polygon meshes and
/// returns the result as connected curves. Use @ref tf::make_form to create
/// forms with the required tree policy (@ref tf::tree or @ref tf::mod_tree)
/// and topology policies (@ref tf::face_membership and @ref tf::manifold_edge_link).
///
/// @tparam Dims The number of dimensions.
/// @tparam Policy0 The policy type for the first mesh form.
/// @tparam Policy1 The policy type for the second mesh form.
/// @param form0 The first mesh @ref tf::form.
/// @param form1 The second mesh @ref tf::form.
/// @return A @ref tf::curves_buffer containing connected intersection curves.
///
/// @see tf::intersections_between_polygons for low-level access.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto make_intersection_curves(const tf::form<Dims, Policy0> &form0,
                              const tf::form<Dims, Policy1> &form1) {
  using Index =
      std::common_type_t<typename Policy0::index_type, typename Policy1::index_type>;
  tf::intersections_between_polygons<Index, double,
                                     tf::coordinate_dims_v<Policy0>>
      ibp;
  ibp.build(form0, form1);
  auto ie = tf::make_intersection_edges(ibp);
  auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
  tf::curves_buffer<Index, tf::coordinate_type<Policy0, Policy1>,
                    tf::coordinate_dims_v<Policy0>>
      cb;
  cb.paths_buffer() = std::move(paths);
  cb.points_buffer().allocate(ibp.intersection_points().size());
  tf::parallel_copy(ibp.intersection_points(), cb.points());
  return cb;
}
} // namespace tf

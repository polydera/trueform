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
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/curves_buffer.hpp"
#include "../cut/cut_graph.hpp"
#include "../cut/dispatch/self_boolean.hpp"
#include "../cut/face_cuts.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./intersections_within_polygons.hpp"
#include "./graph/intersection_graph.hpp"
#include "./intersect_mode.hpp"

namespace tf {

/// @ingroup intersect_curves
/// @brief Extract curves where a mesh intersects itself.
///
/// Finds all locations where a mesh's faces intersect each other
/// (excluding adjacent faces) and returns the result as connected curves.
///
/// @tparam Policy The policy type for the mesh.
/// @param _polygons The input @ref tf::polygons (or tagged form).
/// @param mode The intersection mode (sos or primitives).
/// @return A @ref tf::curves_buffer containing connected self-intersection curves.
template <typename Policy>
auto make_self_intersection_curves(
    const tf::polygons<Policy> &_polygons,
    tf::intersect_mode mode = tf::intersect_mode::sos) {
  return cut::dispatch::self_boolean(_polygons, [mode](const auto &p) {
    using Index = std::decay_t<decltype(p.faces()[0][0])>;
    using RealType = tf::coordinate_type<std::decay_t<decltype(p)>>;

    tf::intersections_within_polygons<Index, RealType> iwp;
    iwp.build(p, mode);

    auto &conv = iwp.converter();
    auto apply_to_face = [&](int, Index object, const auto &f) {
      f(p.faces()[object]);
    };
    auto get_mesh_point = [&](int, Index id) -> tf::point<int32_t, 3> {
      return conv.convert(p.points()[id]);
    };

    tf::intersection_graph<Index> ig;
    ig.build(iwp, apply_to_face, get_mesh_point);

    auto paths = [&]() {
      if (mode == tf::intersect_mode::sos) {
        auto edge_pairs = tf::make_mapped_range(
            ig.edge_groups(),
            [](const auto &group) -> std::array<Index, 2> {
              return {group[0].point_0, group[0].point_1};
            });
        return tf::connect_edges_to_paths(tf::make_edges(edge_pairs));
      } else {
        tf::face_cuts<Index> fc;
        fc.build(ig, apply_to_face, get_mesh_point);

        tf::cut_graph<Index> cg;
        cg.build(fc, ig, p);

        return tf::connect_edges_to_paths(
            tf::make_edges(cg.intersection_edges()));
      }
    }();

    auto ipts = ig.points();
    tf::curves_buffer<Index, RealType, 3> cb;
    cb.paths_buffer() = std::move(paths);
    cb.points_buffer().allocate(ipts.size());
    tf::parallel_copy(
        tf::make_points(tf::make_mapped_range(
            ipts, [&conv](const auto &pt) { return conv.deconvert(pt); })),
        cb.points());
    return cb;
  });
}

} // namespace tf

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
#include "../topology/connect_edges_to_paths.hpp"
#include "./construct/embedded_intersection_curves.hpp"
#include "./dispatch/boolean.hpp"
#include "./dispatch/build_exact_pipeline.hpp"
#include "./return_curves.hpp"

namespace tf {

/// @ingroup cut_boolean
/// @brief Embed intersection curves from mesh B into mesh A.
///
/// Computes intersections between two meshes and embeds the intersection
/// curves into the first mesh's topology. All faces from mesh A are kept
/// (split where intersecting), none from mesh B.
///
/// @tparam Policy0 The policy type of the first mesh.
/// @tparam Policy1 The policy type of the second mesh.
/// @param _polygons0 The mesh to embed curves into.
/// @param _polygons1 The mesh providing the cutting surface.
/// @return A @ref tf::polygons_buffer with embedded intersection edges.
template <typename Int = tf::exact::int32, typename Policy0, typename Policy1>
auto embedded_intersection_curves(const tf::polygons<Policy0> &_polygons0,
                                  const tf::polygons<Policy1> &_polygons1) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        using RealType = tf::coordinate_type<std::decay_t<decltype(p0)>,
                                             std::decay_t<decltype(p1)>>;
        auto [ibp, ig, fc, cg] =
            cut::dispatch::build_exact_pipeline<Index, RealType, Int>(p0, p1);
        return tf::cut::embedded_intersection_curves<Index>(
            p0, ig, fc, ibp.converter(), Index(0));
      });
}

/// @ingroup cut_boolean
/// @brief Embed intersection curves from mesh B into mesh A with curve output.
/// @overload
template <typename Int = tf::exact::int32, typename Policy0, typename Policy1>
auto embedded_intersection_curves(const tf::polygons<Policy0> &_polygons0,
                                  const tf::polygons<Policy1> &_polygons1,
                                  tf::return_curves_t) {
  return cut::dispatch::boolean(
      _polygons0, _polygons1, [](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        using RealType = tf::coordinate_type<std::decay_t<decltype(p0)>,
                                             std::decay_t<decltype(p1)>>;
        auto [ibp, ig, fc, cg] =
            cut::dispatch::build_exact_pipeline<Index, RealType, Int>(p0, p1);
        auto res = tf::cut::embedded_intersection_curves<Index>(
            p0, ig, fc, ibp.converter(), Index(0));

        auto paths =
            tf::connect_edges_to_paths(tf::make_edges(cg.intersection_edges()));
        auto &conv = ibp.converter();
        auto ipts = ig.points();
        tf::curves_buffer<Index, RealType, 3> cb;
        cb.paths_buffer() = std::move(paths);
        cb.points_buffer().allocate(ipts.size());
        tf::parallel_copy(
            tf::make_points(tf::make_mapped_range(
                ipts, [&conv](const auto &pt) { return conv.deconvert(pt); })),
            cb.points());
        return std::make_tuple(std::move(res), std::move(cb));
      });
}

} // namespace tf

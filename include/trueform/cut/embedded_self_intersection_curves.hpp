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
#include "./dispatch/build_self_pipeline.hpp"
#include "./dispatch/self_boolean.hpp"
#include "./return_curves.hpp"

namespace tf {

/// @ingroup cut_boolean
/// @brief Embed self-intersection curves into mesh topology.
///
/// Finds where a mesh intersects itself and splits faces along
/// those curves.
///
/// @tparam Policy The policy type of the mesh.
/// @param _polygons The input @ref tf::polygons (or tagged form).
/// @return A @ref tf::polygons_buffer with embedded intersection edges.
template <typename Int = tf::exact::int32, typename Policy>
auto embedded_self_intersection_curves(const tf::polygons<Policy> &_polygons) {
  return cut::dispatch::self_boolean(_polygons, [](const auto &p) {
    using Index = std::decay_t<decltype(p.faces()[0][0])>;
    using RealType = tf::coordinate_type<std::decay_t<decltype(p)>>;
    auto [iwp, ig, fc, cg] =
        cut::dispatch::build_self_pipeline<Index, RealType, Int>(p);
    return tf::cut::embedded_intersection_curves(p, ig, fc, iwp.converter(),
                                                 Index(0));
  });
}

/// @ingroup cut_boolean
/// @brief Embed self-intersection curves with curve output.
/// @overload
template <typename Int = tf::exact::int32, typename Policy>
auto embedded_self_intersection_curves(const tf::polygons<Policy> &_polygons,
                                       tf::return_curves_t) {
  return cut::dispatch::self_boolean(_polygons, [](const auto &p) {
    using Index = std::decay_t<decltype(p.faces()[0][0])>;
    using RealType = tf::coordinate_type<std::decay_t<decltype(p)>>;
    auto [iwp, ig, fc, cg] =
        cut::dispatch::build_self_pipeline<Index, RealType, Int>(p);
    auto res = tf::cut::embedded_intersection_curves(p, ig, fc, iwp.converter(),
                                                     Index(0));

    auto paths =
        tf::connect_edges_to_paths(tf::make_edges(cg.intersection_edges()));
    auto &conv = iwp.converter();
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

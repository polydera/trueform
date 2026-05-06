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
#include "../exact/resolve_int_type.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./construct/embedded_intersection_curves.hpp"
#include "./dispatch/build_self_pipeline.hpp"
#include "./dispatch/self_boolean.hpp"
#include "./return_curves.hpp"

namespace tf {

/// @ingroup cut_boolean
/// @brief Embed self-intersection curves into mesh topology.
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy>
auto embedded_self_intersection_curves(
    const tf::polygons<Policy> &_polygons,
    tf::intersect_mode mode = tf::intersect_mode::primitives |
                              tf::intersect_mode::resolve_contours) {
  return cut::dispatch::self_boolean(_polygons, [mode](const auto &p) {
    using Index = std::decay_t<decltype(p.faces()[0][0])>;
    using InputReal = tf::coordinate_type<std::decay_t<decltype(p)>>;
    using ResolvedInt = tf::exact::resolve_int_type<Int, InputReal>;
    auto [iwp, ig, fc, cg] =
        cut::dispatch::build_self_pipeline<Index, double, ResolvedInt>(p, mode);
    auto [res, fl] =
        tf::cut::embedded_intersection_curves<Index, OutputCoordinateType>(
            p, ig, fc, iwp.converter(), Index(0));
    return std::make_tuple(std::move(res), std::move(fl));
  });
}

/// @ingroup cut_boolean
/// @brief Embed self-intersection curves with curve output.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy>
auto embedded_self_intersection_curves(const tf::polygons<Policy> &_polygons,
                                       tf::return_curves_t) {
  return embedded_self_intersection_curves<Int, OutputCoordinateType>(
      _polygons, tf::intersect_mode::primitives |
                     tf::intersect_mode::resolve_contours,
      tf::return_curves);
}

/// @ingroup cut_boolean
/// @brief Embed self-intersection curves with curve output.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy>
auto embedded_self_intersection_curves(const tf::polygons<Policy> &_polygons,
                                       tf::intersect_mode mode,
                                       tf::return_curves_t) {
  return cut::dispatch::self_boolean(_polygons, [mode](const auto &p) {
    using Index = std::decay_t<decltype(p.faces()[0][0])>;
    using InputReal = tf::coordinate_type<std::decay_t<decltype(p)>>;
    using ResolvedInt = tf::exact::resolve_int_type<Int, InputReal>;
    using RealOut =
        std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                           InputReal, OutputCoordinateType>;
    auto [iwp, ig, fc, cg] =
        cut::dispatch::build_self_pipeline<Index, double, ResolvedInt>(p, mode);
    auto [res, fl] =
        tf::cut::embedded_intersection_curves<Index, OutputCoordinateType>(
            p, ig, fc, iwp.converter(), Index(0));

    auto paths =
        tf::connect_edges_to_paths(tf::make_edges(cg.intersection_edges()));
    auto &conv = iwp.converter();
    auto ipts = ig.points();
    tf::curves_buffer<Index, RealOut, 3> cb;
    cb.paths_buffer() = std::move(paths);
    cb.points_buffer().allocate(ipts.size());
    tf::parallel_copy(
        tf::make_points(tf::make_mapped_range(
            ipts, [&conv](const auto &pt) { return conv.deconvert(pt); })),
        cb.points());
    return std::make_tuple(std::move(res), std::move(fl), std::move(cb));
  });
}

} // namespace tf

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
#include "../core/none.hpp"
#include "../intersect/make_intersection_edges.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./construct/embedded_isocurves.hpp"
#include "./dispatch/build_iso_pipeline.hpp"
#include "./return_curves.hpp"

namespace tf {

/// @ingroup cut_isocurves
/// @brief Embed scalar field isocurves into mesh topology.
///
/// Creates a new mesh where isocurve contour lines become edges.
/// Faces are split along the curves and labeled by isoband region.
/// Labels indicate which region between consecutive cut values each face
/// belongs to (0 for below first cut, 1 for between first and second, etc.).
///
/// @tparam Index The index type (auto-deduced if not specified).
/// @tparam Policy The policy type of the polygons.
/// @param polygons The input @ref tf::polygons.
/// @param scalars The scalar field values (one per vertex).
/// @param cut_values The threshold values to embed.
/// @return Tuple of (@ref tf::polygons_buffer, labels buffer).
///
/// @see tf::make_isobands for extracting specific bands only.
template <typename Index = tf::none_t, typename Policy, typename Range0,
          typename Iterator0, std::size_t N0>
auto embedded_isocurves(const tf::polygons<Policy> &polygons,
                        const Range0 &scalars,
                        const tf::range<Iterator0, N0> &cut_values) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return embedded_isocurves<ActualIndex>(polygons, scalars, cut_values);
  } else {
    auto [sfi, fc, pids] =
        cut::dispatch::build_iso_pipeline<Index>(polygons, scalars, cut_values);
    auto [res_polygons, labels, face_labels] =
        tf::cut::embedded_isocurves<Index>(polygons, sfi, fc, pids);
    return std::make_tuple(std::move(res_polygons), std::move(labels),
                           std::move(face_labels));
  }
}

/// @ingroup cut_isocurves
/// @brief Embed scalar field isocurves into mesh topology with curve output.
/// @overload
template <typename Index = tf::none_t, typename Policy, typename Range0,
          typename Iterator0, std::size_t N0>
auto embedded_isocurves(const tf::polygons<Policy> &polygons,
                        const Range0 &scalars,
                        const tf::range<Iterator0, N0> &cut_values,
                        tf::return_curves_t) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return embedded_isocurves<ActualIndex>(polygons, scalars, cut_values,
                                           tf::return_curves);
  } else {
    auto [sfi, fc, pids] =
        cut::dispatch::build_iso_pipeline<Index>(polygons, scalars, cut_values);
    auto [res_polygons, labels, face_labels] =
        tf::cut::embedded_isocurves<Index>(polygons, sfi, fc, pids);

    auto ie = tf::make_intersection_edges(sfi, polygons.faces());
    auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
    tf::curves_buffer<Index, tf::coordinate_type<Policy>,
                      tf::coordinate_dims_v<Policy>>
        cb;
    cb.paths_buffer() = std::move(paths);
    cb.points_buffer().allocate(sfi.intersection_points().size());
    tf::parallel_copy(sfi.intersection_points(), cb.points());

    return std::make_tuple(std::move(res_polygons), std::move(labels),
                           std::move(face_labels), std::move(cb));
  }
}

} // namespace tf

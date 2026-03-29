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
#include "../core/none.hpp"
#include "../intersect/make_intersection_edges.hpp"
#include "../reindex/by_ids_on_points.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./construct/make_isobands.hpp"
#include "./dispatch/build_iso_pipeline.hpp"
#include "./return_curves.hpp"

namespace tf {

/// @ingroup cut_isocurves
/// @brief Extract specific isobands from a scalar field.
///
/// Like @ref tf::embedded_isocurves but returns only the selected
/// bands between specified cut values. Useful for extracting regions
/// within specific value ranges.
///
/// @tparam Index The index type (auto-deduced if not specified).
/// @tparam Policy The policy type of the polygons.
/// @param polygons The input @ref tf::polygons.
/// @param scalars The scalar field values (one per vertex).
/// @param cut_values The threshold values defining band boundaries.
/// @param selected_bands Indices of bands to extract.
/// @return Tuple of (@ref tf::polygons_buffer, labels buffer).
///
/// @see tf::embedded_isocurves for embedding all bands.
template <typename Index = tf::none_t, typename Policy, typename Range0,
          typename Iterator0, std::size_t N0, typename Iterator1,
          std::size_t N1>
auto make_isobands(const tf::polygons<Policy> &polygons, const Range0 &scalars,
                   const tf::range<Iterator0, N0> &cut_values,
                   const tf::range<Iterator1, N1> &selected_bands) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return make_isobands<ActualIndex>(polygons, scalars, cut_values,
                                      selected_bands);
  } else {
    auto [sfi, fc, pids] =
        cut::dispatch::build_iso_pipeline<Index>(polygons, scalars, cut_values);
    auto [res_polygons, labels, created_ids] =
        tf::cut::make_isobands<Index>(polygons, sfi, fc, pids, selected_bands);
    return std::make_pair(std::move(res_polygons), std::move(labels));
  }
}

/// @ingroup cut_isocurves
/// @brief Extract specific isobands from a scalar field with curve output.
/// @overload
template <typename Index = tf::none_t, typename Policy, typename Range0,
          typename Iterator0, std::size_t N0, typename Iterator1,
          std::size_t N1>
auto make_isobands(const tf::polygons<Policy> &polygons, const Range0 &scalars,
                   const tf::range<Iterator0, N0> &cut_values,
                   const tf::range<Iterator1, N1> &selected_bands,
                   tf::return_curves_t) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return make_isobands<ActualIndex>(polygons, scalars, cut_values,
                                      selected_bands, tf::return_curves);
  } else {
    auto [sfi, fc, pids] =
        cut::dispatch::build_iso_pipeline<Index>(polygons, scalars, cut_values);
    auto [res_polygons, labels, created_ids] =
        tf::cut::make_isobands<Index>(polygons, sfi, fc, pids, selected_bands);

    auto ie = tf::make_intersection_edges(sfi);
    auto all_segments =
        tf::make_segments(tf::make_edges(ie), sfi.intersection_points());
    auto filtered_segments =
        tf::reindexed_by_ids_on_points(all_segments, created_ids);
    auto paths = tf::connect_edges_to_paths(filtered_segments.edges());

    tf::curves_buffer<Index, tf::coordinate_type<Policy>,
                      tf::coordinate_dims_v<Policy>>
        cb;
    cb.paths_buffer() = std::move(paths);
    cb.points_buffer() = std::move(filtered_segments.points_buffer());

    return std::make_tuple(std::move(res_polygons), std::move(labels),
                           std::move(cb));
  }
}

} // namespace tf

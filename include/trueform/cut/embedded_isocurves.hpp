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
#include "../intersect/scalar_field_intersections.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./impl/embedded_isocurves.hpp"
#include "./return_curves.hpp"
#include "./scalar_cut_faces.hpp"

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
/// @see tf::make_isocontours for curve extraction without mesh cutting.
template <typename Index = tf::none_t, typename Policy, typename Range0,
          typename Iterator0, std::size_t N0>
auto embedded_isocurves(const tf::polygons<Policy> &polygons,
                        const Range0 &scalars,
                        const tf::range<Iterator0, N0> &cut_values) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return embedded_isocurves<ActualIndex>(polygons, scalars, cut_values);
  } else {
    tf::buffer<std::decay_t<decltype(cut_values[0])>> cut_vals;
    cut_vals.reserve(cut_values.size());
    std::copy(cut_values.begin(), cut_values.end(),
              std::back_inserter(cut_vals));
    std::sort(cut_vals.begin(), cut_vals.end());
    tf::scalar_field_intersections<Index, tf::coordinate_type<Policy>,
                                   tf::coordinate_dims_v<Policy>>
        sfi;
    sfi.build_many(polygons, scalars, cut_vals);
    tf::scalar_cut_faces<Index> scf;
    scf.build(polygons, sfi);
    return cut::embedded_isocurves<Index>(polygons, sfi, scf, scalars,
                                          tf::make_range(cut_vals));
  }
}

/// @ingroup cut_isocurves
/// @brief Embed scalar field isocurves into mesh topology with curve output.
/// @overload
///
/// @param polygons The input @ref tf::polygons.
/// @param scalars The scalar field values (one per vertex).
/// @param cut_values The threshold values to embed.
/// @param tag Pass @ref tf::return_curves to get curves.
/// @return Tuple of (@ref tf::polygons_buffer, labels buffer, @ref tf::curves_buffer).
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
    tf::buffer<std::decay_t<decltype(cut_values[0])>> cut_vals;
    cut_vals.reserve(cut_values.size());
    std::copy(cut_values.begin(), cut_values.end(),
              std::back_inserter(cut_vals));
    std::sort(cut_vals.begin(), cut_vals.end());
    tf::scalar_field_intersections<Index, tf::coordinate_type<Policy>,
                                   tf::coordinate_dims_v<Policy>>
        sfi;
    sfi.build_many(polygons, scalars, cut_vals);
    tf::scalar_cut_faces<Index> scf;
    scf.build(polygons, sfi);
    auto [res_polygons, labels] = cut::embedded_isocurves<Index>(
        polygons, sfi, scf, scalars, tf::make_range(cut_vals));
    auto ie = tf::make_mapped_range(scf.intersection_edges(), [](auto e) {
      return std::array<Index, 2>{e[0].id, e[1].id};
    });
    auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
    tf::curves_buffer<Index, tf::coordinate_type<Policy>,
                      tf::coordinate_dims_v<Policy>>
        cb;
    cb.paths_buffer() = std::move(paths);
    cb.points_buffer().allocate(sfi.intersection_points().size());
    tf::parallel_copy(sfi.intersection_points(), cb.points());
    return std::make_tuple(std::move(res_polygons), std::move(labels),
                           std::move(cb));
  }
}
} // namespace tf

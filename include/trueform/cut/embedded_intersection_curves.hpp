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
#include "../intersect/intersections_between_polygons.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./impl/dispatch.hpp"
#include "./impl/embedded_intersection_curves.hpp"
#include "./return_curves.hpp"
#include "./tagged_cut_faces.hpp"

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
template <typename Policy0, typename Policy1>
auto embedded_intersection_curves(const tf::polygons<Policy0> &_polygons0,
                                  const tf::polygons<Policy1> &_polygons1) {
  return cut::impl::boolean_dispatch(
      _polygons0, _polygons1, [](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        tf::intersections_between_polygons<Index, double, 3> ibp;
        ibp.build(p0, p1);
        tf::tagged_cut_faces<Index> tcf;
        tcf.build(p0, p1, ibp);
        return tf::cut::embedded_intersection_curves<Index>(p0, ibp, tcf);
      });
}

/// @ingroup cut_boolean
/// @brief Embed intersection curves from mesh B into mesh A with curve output.
/// @overload
///
/// @param _polygons0 The mesh to embed curves into.
/// @param _polygons1 The mesh providing the cutting surface.
/// @param tag Pass @ref tf::return_curves to get curves.
/// @return Tuple of (@ref tf::polygons_buffer, @ref tf::curves_buffer).
template <typename Policy0, typename Policy1>
auto embedded_intersection_curves(const tf::polygons<Policy0> &_polygons0,
                                  const tf::polygons<Policy1> &_polygons1,
                                  tf::return_curves_t) {
  return cut::impl::boolean_dispatch(
      _polygons0, _polygons1, [](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        using RealT = tf::coordinate_type<std::decay_t<decltype(p0)>>;
        tf::intersections_between_polygons<Index, double, 3> ibp;
        ibp.build(p0, p1);
        tf::tagged_cut_faces<Index> tcf;
        tcf.build(p0, p1, ibp);
        auto res = tf::cut::embedded_intersection_curves<Index>(p0, ibp, tcf);
        auto ie = tf::make_mapped_range(tcf.intersection_edges(), [](auto e) {
          return std::array<Index, 2>{e[0].id, e[1].id};
        });
        auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
        tf::curves_buffer<Index, RealT, 3> cb;
        cb.paths_buffer() = std::move(paths);
        cb.points_buffer().allocate(ibp.intersection_points().size());
        tf::parallel_copy(ibp.intersection_points(), cb.points());
        return std::make_tuple(std::move(res), std::move(cb));
      });
}

} // namespace tf

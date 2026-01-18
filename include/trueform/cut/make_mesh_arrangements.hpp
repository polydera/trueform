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
#include "../intersect/intersections_between_polygons.hpp"
#include "./impl/dispatch.hpp"
#include "./impl/make_mesh_arrangements.hpp"
#include "./tagged_cut_faces.hpp"

namespace tf {

/// @ingroup cut_boolean
/// @brief Decompose two meshes into classified regions.
///
/// Returns all subdivided regions created by mesh intersection,
/// each classified by origin and spatial relationship (inside/outside).
/// This is the complete decomposition from which any boolean operation
/// can be reconstructed.
///
/// @tparam Policy0 The policy type of the first mesh.
/// @tparam Policy1 The policy type of the second mesh.
/// @param _polygons0 The first mesh @ref tf::polygons (or tagged form).
/// @param _polygons1 The second mesh @ref tf::polygons (or tagged form).
/// @return Tuple of (vector of @ref tf::polygons_buffer, labels, @ref tf::arrangement_class).
///
/// @see tf::make_boolean for combined boolean results.
/// @see tf::arrangement_class for classification values.
template <typename Policy0, typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1) {
  return cut::impl::boolean_dispatch(
      _polygons0, _polygons1, [](const auto &p0, const auto &p1) {
        using Index = std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                                         typename std::decay_t<decltype(p1)>::index_type>;
        tf::intersections_between_polygons<Index, double, 3> ibp;
        ibp.build(p0, p1);
        tf::tagged_cut_faces<Index> tcf;
        tcf.build(p0, p1, ibp);
        return tf::cut::make_mesh_arrangements<int>(p0, p1, ibp, tcf);
      });
}

} // namespace tf

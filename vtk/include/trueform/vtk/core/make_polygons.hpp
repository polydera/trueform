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
#include <trueform/vtk/core/make_points.hpp>
#include <trueform/vtk/core/make_polys.hpp>
#include <vtkPolyData.h>

namespace tf::vtk {

template <std::size_t V>
using polygons_sized_t = decltype(tf::make_polygons(
    std::declval<polys_sized_t<V>>(), std::declval<points_t>()));

using polygons_t = polygons_sized_t<tf::dynamic_size>;

/// @brief Creates a sized polygons view from vtkPolyData (zero-copy).
/// @tparam V Vertex count per polygon (e.g., 3 for triangles).
/// @param poly VTK poly data object, may be nullptr.
/// @return A tf::polygons view over the underlying data.
template <std::size_t V>
auto make_polygons(vtkPolyData *poly) -> polygons_sized_t<V> {
  return tf::make_polygons(make_polys<V>(poly ? poly->GetPolys() : nullptr),
                           make_points(poly));
}

/// @brief Creates a polygons view from vtkPolyData (zero-copy).
/// @param poly VTK poly data object, may be nullptr.
/// @return A tf::polygons view over the underlying data.
auto make_polygons(vtkPolyData *poly) -> polygons_t;

template <>
inline auto make_polygons<tf::dynamic_size>(vtkPolyData *poly) -> polygons_t {
  return make_polygons(poly);
}

extern template auto make_polygons<3>(vtkPolyData *poly) -> polygons_sized_t<3>;

} // namespace tf::vtk

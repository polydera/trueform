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
#include <trueform/core/curves_buffer.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/segments_buffer.hpp>
#include <trueform/vtk/core/make_vtk_cells.hpp>
#include <trueform/vtk/core/make_vtk_points.hpp>
#include <vtkPolyData.h>

namespace tf::vtk {

/// @brief Creates vtkPolyData from a polygons_buffer (copies data).
/// @tparam V Vertex count per polygon.
/// @param polys Trueform polygons buffer.
/// @return A new vtkPolyData with copied data.
template <std::size_t V>
auto make_vtk_polydata(const tf::polygons_buffer<vtkIdType, float, 3, V> &polys)
    -> vtkSmartPointer<vtkPolyData> {
  auto out = vtkSmartPointer<vtkPolyData>::New();
  out->SetPoints(make_vtk_points(polys.points_buffer()));
  out->SetPolys(make_vtk_cells(polys.faces_buffer()));
  return out;
}

/// @brief Creates vtkPolyData from a polygons_buffer (moves data, zero-copy).
/// @tparam V Vertex count per polygon.
/// @param polys Trueform polygons buffer with vtkIdType indices (consumed).
/// @return A new vtkPolyData with transferred ownership.
template <std::size_t V>
auto make_vtk_polydata(tf::polygons_buffer<vtkIdType, float, 3, V> &&polys)
    -> vtkSmartPointer<vtkPolyData> {
  auto out = vtkSmartPointer<vtkPolyData>::New();
  out->SetPoints(make_vtk_points(std::move(polys.points_buffer())));
  out->SetPolys(make_vtk_cells(std::move(polys.faces_buffer())));
  return out;
}

/// @brief Creates vtkPolyData from a dynamic polygons_buffer (copies data).
/// @param polys Trueform polygons buffer.
/// @return A new vtkPolyData with copied data.
auto make_vtk_polydata(
    const tf::polygons_buffer<vtkIdType, float, 3, tf::dynamic_size> &polys)
    -> vtkSmartPointer<vtkPolyData>;

/// @brief Creates vtkPolyData from a dynamic polygons_buffer (moves data,
/// zero-copy).
/// @param polys Trueform polygons buffer with vtkIdType indices (consumed).
/// @return A new vtkPolyData with transferred ownership.
auto make_vtk_polydata(
    tf::polygons_buffer<vtkIdType, float, 3, tf::dynamic_size> &&polys)
    -> vtkSmartPointer<vtkPolyData>;

/// @brief Creates vtkPolyData (lines) from a curves_buffer (copies data).
/// @param curves Trueform curves buffer.
/// @return A new vtkPolyData with lines.
auto make_vtk_polydata(const tf::curves_buffer<vtkIdType, float, 3> &curves)
    -> vtkSmartPointer<vtkPolyData>;

/// @brief Creates vtkPolyData (lines) from a curves_buffer (moves data,
/// zero-copy).
/// @param curves Trueform curves buffer with vtkIdType indices (consumed).
/// @return A new vtkPolyData with lines.
auto make_vtk_polydata(tf::curves_buffer<vtkIdType, float, 3> &&curves)
    -> vtkSmartPointer<vtkPolyData>;

/// @brief Creates vtkPolyData (lines) from a segments_buffer (copies data).
/// @param segments Trueform segments buffer.
/// @return A new vtkPolyData with lines.
auto make_vtk_polydata(const tf::segments_buffer<vtkIdType, float, 3> &segments)
    -> vtkSmartPointer<vtkPolyData>;

/// @brief Creates vtkPolyData (lines) from a segments_buffer (moves data,
/// zero-copy).
/// @param segments Trueform segments buffer with vtkIdType indices (consumed).
/// @return A new vtkPolyData with lines.
auto make_vtk_polydata(tf::segments_buffer<vtkIdType, float, 3> &&segments)
    -> vtkSmartPointer<vtkPolyData>;

} // namespace tf::vtk

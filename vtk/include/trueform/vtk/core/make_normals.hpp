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
#include <trueform/core.hpp>

class vtkPolyData;

namespace tf::vtk {

/// @brief Unit vectors view type for VTK float normals.
using normals_t = decltype(tf::make_unit_vectors<3>(
    tf::make_range(static_cast<float *>(nullptr), std::size_t{0})));

/// @brief Creates a unit vectors view from vtkPolyData point normals (zero-copy).
/// @param poly VTK poly data object, may be nullptr.
/// @return A tf::unit_vectors view over point normals, or empty if nullptr/no normals.
auto make_point_normals(vtkPolyData *poly) -> normals_t;

/// @brief Creates a unit vectors view from vtkPolyData cell normals (zero-copy).
/// @param poly VTK poly data object, may be nullptr.
/// @return A tf::unit_vectors view over cell normals, or empty if nullptr/no normals.
auto make_cell_normals(vtkPolyData *poly) -> normals_t;

} // namespace tf::vtk

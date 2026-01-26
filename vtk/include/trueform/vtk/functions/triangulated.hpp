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
#include <vtkSmartPointer.h>

namespace tf::vtk {

class polydata;

/// @brief Triangulate all polygons in the mesh.
/// @param input The polydata to triangulate.
/// @param preserve_point_data If true, copy point data arrays (default: true).
/// @return A new polydata with triangulated faces.
/// @note Cell data is not preserved since face count changes.
auto triangulated(polydata *input, bool preserve_point_data = true)
    -> vtkSmartPointer<polydata>;

} // namespace tf::vtk

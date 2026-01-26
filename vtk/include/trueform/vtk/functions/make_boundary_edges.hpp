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
#include <trueform/vtk/core/polydata.hpp>
#include <vtkSmartPointer.h>

namespace tf::vtk {

/// @brief Extract boundary edges from a mesh.
///
/// Returns a polydata containing line cells representing edges that
/// belong to only one face. The output shares points with the input.
///
/// @param input The input mesh (must be tf::vtk::polydata).
/// @return Polydata with boundary edges as lines.
///
/// @code
/// auto edges = tf::vtk::make_boundary_edges(mesh);
/// std::cout << "Found " << edges->GetNumberOfLines() << " boundary edges\n";
/// @endcode
auto make_boundary_edges(polydata *input) -> vtkSmartPointer<polydata>;

} // namespace tf::vtk

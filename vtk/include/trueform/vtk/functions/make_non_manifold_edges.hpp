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

/// @brief Extract non-manifold edges from a mesh.
///
/// Returns a polydata containing line cells representing edges that
/// belong to more than two faces. The output shares points with the input.
///
/// @param input The input mesh (must be tf::vtk::polydata).
/// @return Polydata with non-manifold edges as lines.
///
/// @code
/// auto edges = tf::vtk::make_non_manifold_edges(mesh);
/// std::cout << "Found " << edges->GetNumberOfLines() << " non-manifold edges\n";
/// @endcode
auto make_non_manifold_edges(polydata *input) -> vtkSmartPointer<polydata>;

} // namespace tf::vtk

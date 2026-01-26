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
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

namespace tf::vtk {

/// @brief Extract boundary paths from a mesh.
///
/// Returns a polydata containing line cells representing boundary edges
/// connected into paths. The output shares points with the input.
///
/// @param input The input mesh (must be tf::vtk::polydata).
/// @return Polydata with boundary paths as lines.
///
/// @code
/// vtkNew<tf::vtk::adapter> adapter;
/// adapter->SetInputConnection(reader->GetOutputPort());
/// adapter->Update();
///
/// auto boundaries = tf::vtk::make_boundary_paths(adapter->GetOutput());
/// std::cout << "Found " << boundaries->GetNumberOfLines() << " boundary paths\n";
/// @endcode
auto make_boundary_paths(polydata *input) -> vtkSmartPointer<polydata>;

} // namespace tf::vtk

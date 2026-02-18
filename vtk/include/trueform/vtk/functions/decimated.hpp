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
#include <trueform/remesh/decimate_config.hpp>
#include <vtkSmartPointer.h>

namespace tf::vtk {

class polydata;

/// @brief Decimate a mesh using quadric error metric.
/// @param input The polydata to decimate. Assumes triangles.
/// @param target_proportion Fraction of faces to keep (0 to 1).
/// @param config Decimation configuration.
/// @return A new polydata with the decimated mesh and cached half-edges.
auto decimated(polydata *input, float target_proportion,
               const tf::decimate_config<float> &config = {})
    -> vtkSmartPointer<polydata>;

} // namespace tf::vtk

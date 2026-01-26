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

class vtkPoints;

namespace tf::vtk {

/// @brief Cleans points by removing duplicates.
/// @param input The points to clean.
/// @param tolerance Distance tolerance for merging points (0 = exact duplicates only).
/// @return A new vtkPoints with duplicates removed.
auto cleaned_points(vtkPoints *input, float tolerance = 0.f)
    -> vtkSmartPointer<vtkPoints>;

} // namespace tf::vtk

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
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

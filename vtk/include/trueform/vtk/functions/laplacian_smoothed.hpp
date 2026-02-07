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

class polydata;

/// @brief Apply Laplacian smoothing to mesh vertices.
///
/// Iteratively moves each vertex towards the centroid of its neighbors.
/// The amount of movement is controlled by lambda (0 = no movement, 1 = full).
///
/// @param input The polydata to smooth.
/// @param iterations Number of smoothing iterations.
/// @param lambda Smoothing factor in [0, 1]. Default: 0.5.
/// @return New vtkPoints with smoothed positions.
///
/// @code
/// auto smoothed_points = tf::vtk::laplacian_smoothed(poly, 10, 0.5f);
/// poly->SetPoints(smoothed_points);
/// @endcode
auto laplacian_smoothed(polydata *input, std::size_t iterations,
                        float lambda = 0.5f) -> vtkSmartPointer<vtkPoints>;

} // namespace tf::vtk

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

/// @brief Apply Taubin smoothing to mesh vertices.
///
/// Taubin smoothing alternates between shrinking (positive lambda) and
/// inflating (negative mu) passes to smooth without significant volume loss.
/// This avoids the shrinkage problem of standard Laplacian smoothing.
///
/// The mu parameter is computed internally from lambda and the pass-band
/// frequency kpb: mu = 1 / (kpb - 1/lambda)
///
/// With lambda=0.5 and kpb=0.1, mu ≈ -0.526.
///
/// @param input The polydata to smooth.
/// @param iterations Number of smoothing iterations (each iteration is one
///                   shrink + one inflate pass).
/// @param lambda Smoothing factor for shrink pass. Default: 0.5.
/// @param kpb Pass-band frequency. Default: 0.1.
/// @return New vtkPoints with smoothed positions.
///
/// @code
/// auto smoothed_points = tf::vtk::taubin_smoothed(poly, 10, 0.5f, 0.1f);
/// poly->SetPoints(smoothed_points);
/// @endcode
auto taubin_smoothed(polydata *input, std::size_t iterations,
                     float lambda = 0.5f, float kpb = 0.1f)
    -> vtkSmartPointer<vtkPoints>;

} // namespace tf::vtk

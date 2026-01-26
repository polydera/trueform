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

/// @brief Cleans polygons by removing duplicate points and faces, and degenerate faces.
/// Returns a new polydata with all point and cell data arrays remapped.
/// @param input The polydata to clean.
/// @param tolerance Distance tolerance for merging points (0 = exact duplicates only).
/// @param preserve_data If true, remap point and cell data arrays (default: true).
/// @return A new polydata with cleaned polygons.
auto cleaned_polygons(polydata *input, float tolerance = 0.f,
                      bool preserve_data = true) -> vtkSmartPointer<polydata>;

} // namespace tf::vtk

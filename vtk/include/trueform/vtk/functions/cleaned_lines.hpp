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

/// @brief Cleans lines by removing duplicate points, degenerate edges,
/// and reconnecting edges into continuous paths.
///
/// This function:
/// 1. Extracts all edges from lines
/// 2. Cleans segments (merge duplicate points, remove degenerate edges)
/// 3. Reconnects edges into continuous paths
///
/// Note: Cell data cannot be preserved because edges are reconnected into paths.
///
/// @param input The polydata with lines to clean.
/// @param tolerance Distance tolerance for merging points (0 = exact duplicates only).
/// @param preserve_data If true, remap point data arrays (default: true).
/// @return A new polydata with cleaned and reconnected lines.
auto cleaned_lines(polydata *input, float tolerance = 0.f,
                   bool preserve_data = true) -> vtkSmartPointer<polydata>;

} // namespace tf::vtk

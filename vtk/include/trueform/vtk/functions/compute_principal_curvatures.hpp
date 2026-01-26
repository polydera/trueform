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
* Author: Ziga Sajovic
*/
#pragma once

namespace tf::vtk {

class polydata;

/// @brief Compute principal curvatures and set them on the polydata.
/// @param input The polydata (must contain 3D polygons).
/// @param k Number of rings for neighborhood (default 2).
/// @param compute_directions If true, also compute principal directions.
/// @note Sets "K1", "K2" scalar arrays on point data.
/// @note If compute_directions is true, also sets "D1", "D2" vector arrays.
auto compute_principal_curvatures(polydata *input, int k = 2,
                                  bool compute_directions = false) -> void;

} // namespace tf::vtk

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

namespace tf::vtk {

class polydata;

/// @brief Ensure faces are oriented with outward-pointing normals.
/// @param input The polydata (must contain 3D polygons).
/// @param is_consistent If true, skip consistency step (default: false).
/// @note First orients faces consistently, then flips all if signed volume is
/// negative.
auto ensure_positive_orientation(polydata *input, bool is_consistent = false)
    -> void;

} // namespace tf::vtk

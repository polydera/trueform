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
/// @return `true` when the mesh is now consistent and positively oriented.
/// @note First orients faces consistently, then flips all if signed volume is
/// negative. A surface carrying a non-orientable component has no outward
/// side, so the volume flip is not taken.
auto ensure_positive_orientation(polydata *input, bool is_consistent = false)
    -> bool;

} // namespace tf::vtk

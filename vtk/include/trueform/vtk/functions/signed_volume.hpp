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

/// @brief Compute the signed volume of a closed mesh.
/// @param input The polydata (must contain 3D polygons).
/// @return Signed volume (positive = outward normals, negative = inward).
auto signed_volume(polydata *input) -> double;

} // namespace tf::vtk

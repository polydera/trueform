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

/// @brief Orient faces consistently so adjacent faces have compatible winding.
/// @param input The polydata (must contain 3D polygons).
/// @note Uses flood-fill through manifold edges. Non-manifold edges act as
/// barriers. The final orientation preserves the majority area within each
/// region.
auto orient_faces_consistently(polydata *input) -> void;

} // namespace tf::vtk

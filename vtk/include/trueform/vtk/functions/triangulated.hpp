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

/// @brief Triangulate all polygons in the mesh.
///
/// A face whose loop crosses itself is resolved rather than dropped: it states
/// its crossing and mints the identity that names it. The output shares the
/// input's points whenever nothing was minted — which is the mesh in practice
/// — and owns its own table otherwise.
///
/// @param input The polydata to triangulate.
/// @param preserve_point_data If true, copy point data arrays (default: true).
/// @return A new polydata with triangulated faces.
/// @note Cell data is not preserved since face count changes. Point data is
///       preserved only when the points are the input's own: a resolution
///       appends identities the input's arrays have no row for.
auto triangulated(polydata *input, bool preserve_point_data = true)
    -> vtkSmartPointer<polydata>;

} // namespace tf::vtk

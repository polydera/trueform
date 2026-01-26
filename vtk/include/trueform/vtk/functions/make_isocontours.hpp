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
#include <trueform/vtk/core/polydata.hpp>
#include <vector>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

namespace tf::vtk {

/// @brief Extract isocontours at specified scalar values.
///
/// Returns a polydata containing line cells representing isocontours
/// at the given cut values.
///
/// @param input The input mesh with float point scalars.
/// @param scalars_name Name of the float point data array to use (nullptr for active scalars).
/// @param values The scalar values for the isocontours.
/// @return Polydata with isocontour curves as lines, or nullptr on failure.
///
/// @code
/// auto contours = tf::vtk::make_isocontours(mesh, "height", {0.1f, 0.2f, 0.3f});
/// @endcode
auto make_isocontours(vtkPolyData *input, const char *scalars_name,
                      const std::vector<float> &values)
    -> vtkSmartPointer<polydata>;

} // namespace tf::vtk

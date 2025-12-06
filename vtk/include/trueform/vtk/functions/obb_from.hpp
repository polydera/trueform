/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/core.hpp>

class vtkPolyData;

namespace tf::vtk {

/// @brief Compute oriented bounding box from polydata points.
/// @param input The vtkPolyData.
/// @return The oriented bounding box.
auto obb_from(vtkPolyData *input) -> tf::obb<float, 3>;

} // namespace tf::vtk

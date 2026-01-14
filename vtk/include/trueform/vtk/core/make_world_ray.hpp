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
#include <trueform/core.hpp>
#include <vtkRenderer.h>

namespace tf::vtk {

/// @brief Get a world-space ray from screen coordinates.
/// @param renderer The renderer.
/// @param x Screen x coordinate.
/// @param y Screen y coordinate.
/// @return Ray in world coordinates.
auto make_world_ray(vtkRenderer *renderer, int x, int y) -> tf::ray<float, 3>;

} // namespace tf::vtk

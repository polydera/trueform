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
#include <trueform/spatial/nearest_neighbor.hpp>
#include <trueform/spatial/nearest_neighbor_pair.hpp>
#include <vtkType.h>

namespace tf::vtk {

/// @brief Result of a neighbor search (form vs primitive).
/// Has .element (vtkIdType), .info.metric (squared distance), .info.point (tf::point<float, 3>).
/// Convertible to bool.
using neighbor_result = tf::nearest_neighbor<vtkIdType, float, 3>;

/// @brief Result of a neighbor search (form vs form).
/// Has .elements (pair of vtkIdType), .info.metric (squared distance), .info.first, .info.second (tf::point<float, 3>).
/// Convertible to bool.
using neighbor_pair_result = tf::nearest_neighbor_pair<vtkIdType, vtkIdType, float, 3>;

} // namespace tf::vtk

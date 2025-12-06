/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <trueform/core.hpp>
#include <vtkType.h>

namespace tf::vtk {

/// @brief Result of a neighbor search (form vs primitive).
struct neighbor_result {
  vtkIdType cell_id;
  float distance;
  tf::point<float, 3> closest_point;
};

/// @brief Result of a neighbor search (form vs form).
struct neighbor_pair_result {
  vtkIdType cell_id0;
  vtkIdType cell_id1;
  float distance;
  tf::point<float, 3> point0;
  tf::point<float, 3> point1;
};

} // namespace tf::vtk

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
#include <trueform/core/area.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/area.hpp>

namespace tf::vtk {

auto area(polydata *input) -> double {
  if (!input || input->GetNumberOfPolys() == 0) {
    return 0.0;
  }

  return tf::area(
      tf::make_polygons(input->polys(), input->points().as<double>()));
}

} // namespace tf::vtk

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
#include <trueform/core/polygons.hpp>
#include <trueform/geometry/ensure_positive_orientation.hpp>
#include <trueform/topology.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/ensure_positive_orientation.hpp>

namespace tf::vtk {

auto ensure_positive_orientation(polydata *input, bool is_consistent) -> bool {
  if (!input) {
    return false;
  }
  if (input->GetNumberOfPolys() == 0) {
    return true;
  }

  // The signed volume decides a global flip, so it is read in double even
  // though the stored coordinates are float.
  auto polygons =
      tf::make_polygons(input->polys(), input->points().as<double>()) |
      tf::tag(input->manifold_edge_link());

  const bool oriented = tf::ensure_positive_orientation(polygons, is_consistent);
  input->GetPolys()->Modified();
  return oriented;
}

} // namespace tf::vtk

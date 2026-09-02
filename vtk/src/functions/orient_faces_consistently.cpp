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
#include <trueform/topology.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/orient_faces_consistently.hpp>

namespace tf::vtk {

auto orient_faces_consistently(polydata *input) -> bool {
  if (!input) {
    return false;
  }

  auto polygons = input->polygons() |
                  tf::tag(input->manifold_edge_link());

  const bool oriented = tf::orient_faces_consistently(polygons);
  input->GetPolys()->Modified();
  return oriented;
}

} // namespace tf::vtk

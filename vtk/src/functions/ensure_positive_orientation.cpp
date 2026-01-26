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
#include <trueform/topology/reverse_winding.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/ensure_positive_orientation.hpp>
#include <trueform/vtk/functions/orient_faces_consistently.hpp>
#include <trueform/vtk/functions/signed_volume.hpp>

namespace tf::vtk {

auto ensure_positive_orientation(polydata *input, bool is_consistent) -> void {
  if (!input || input->GetNumberOfPolys() == 0) {
    return;
  }

  if (!is_consistent) {
    orient_faces_consistently(input);
  }

  if (signed_volume(input) < 0) {
    tf::reverse_winding(input->polygons().faces());
    input->GetPolys()->Modified();
  }
}

} // namespace tf::vtk

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/core/signed_volume.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/signed_volume.hpp>

namespace tf::vtk {

auto signed_volume(polydata *input) -> double {
  if (!input || input->GetNumberOfPolys() == 0) {
    return 0.0;
  }

  return tf::signed_volume(input->polygons());
}

} // namespace tf::vtk

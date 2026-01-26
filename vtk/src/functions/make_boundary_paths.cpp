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
#include <trueform/vtk/functions/make_boundary_paths.hpp>
#include <trueform/topology/boundary_paths.hpp>
#include <trueform/vtk/core/make_vtk_cells.hpp>

namespace tf::vtk {

auto make_boundary_paths(polydata *input) -> vtkSmartPointer<polydata> {
  if (!input) {
    return nullptr;
  }

  auto out = vtkSmartPointer<polydata>::New();

  // Share points with input
  out->SetPoints(input->GetPoints());

  auto paths = tf::make_boundary_paths(
      input->polygons() | tf::tag(input->face_membership()));
  out->SetLines(make_vtk_cells(std::move(paths)));

  return out;
}

} // namespace tf::vtk

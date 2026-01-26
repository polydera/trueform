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
#include <trueform/vtk/functions/make_non_manifold_edges.hpp>
#include <trueform/topology/non_simple_edges.hpp>
#include <trueform/vtk/core/make_vtk_cells.hpp>

namespace tf::vtk {

auto make_non_manifold_edges(polydata *input) -> vtkSmartPointer<polydata> {
  if (!input) {
    return nullptr;
  }

  auto [boundary, non_manifold] = tf::make_non_simple_edges(
      input->polygons() | tf::tag(input->face_membership()));

  auto out = vtkSmartPointer<polydata>::New();
  out->SetPoints(input->GetPoints());
  out->SetLines(make_vtk_cells(std::move(non_manifold)));

  return out;
}

} // namespace tf::vtk

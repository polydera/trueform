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
#include <trueform/vtk/functions/compute_point_normals.hpp>
#include <trueform/vtk/functions/compute_cell_normals.hpp>
#include <trueform/geometry.hpp>
#include <trueform/vtk/core/make_normals.hpp>
#include <trueform/vtk/core/make_vtk_normals.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>

namespace tf::vtk {

auto compute_point_normals(polydata *input) -> void {
  if (!input) {
    return;
  }

  // Compute cell normals if not present
  if (input->cell_normals().size() == 0) {
    compute_cell_normals(input);
  }

  auto polygons = input->polygons() | tf::tag(input->face_membership())
                  | tf::tag_normals(input->cell_normals());

  auto normals = tf::compute_point_normals(polygons);
  auto vtk_normals = make_vtk_normals(std::move(normals));
  vtk_normals->SetName("Normals");
  input->GetPointData()->SetNormals(vtk_normals);
}

} // namespace tf::vtk

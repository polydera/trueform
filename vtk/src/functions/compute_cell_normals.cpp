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
#include <trueform/vtk/functions/compute_cell_normals.hpp>
#include <trueform/geometry.hpp>
#include <trueform/vtk/core/make_polygons.hpp>
#include <trueform/vtk/core/make_vtk_normals.hpp>
#include <vtkCellData.h>
#include <vtkPolyData.h>
#include <vtkFloatArray.h>

namespace tf::vtk {

auto compute_cell_normals(vtkPolyData *input) -> void {
  if (!input) {
    return;
  }

  auto polygons = make_polygons(input);
  auto normals = tf::compute_normals(polygons);
  auto vtk_normals = make_vtk_normals(std::move(normals));
  vtk_normals->SetName("Normals");
  input->GetCellData()->SetNormals(vtk_normals);
}

} // namespace tf::vtk

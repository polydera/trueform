/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
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

  auto polygons = make_polygons<3>(input);
  auto normals = tf::compute_normals(polygons);
  auto vtk_normals = make_vtk_normals(std::move(normals));
  vtk_normals->SetName("Normals");
  input->GetCellData()->SetNormals(vtk_normals);
}

} // namespace tf::vtk

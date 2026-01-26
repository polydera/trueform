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
#include <trueform/geometry/triangulated_faces.hpp>
#include <trueform/vtk/core/make_vtk_cells.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/triangulated.hpp>
#include <vtkPointData.h>
#include <vtkPoints.h>

namespace tf::vtk {

auto triangulated(polydata *input, bool preserve_point_data)
    -> vtkSmartPointer<polydata> {
  if (!input || input->GetNumberOfPolys() == 0) {
    return nullptr;
  }

  auto triangle_faces = tf::triangulated_faces(input->polygons());

  auto out = vtkSmartPointer<polydata>::New();
  out->Initialize();

  // Points are unchanged - share them
  out->SetPoints(input->GetPoints());

  // Set triangulated polys
  out->SetPolys(make_vtk_cells(std::move(triangle_faces)));

  if (preserve_point_data && input->GetPointData() &&
      input->GetPointData()->GetNumberOfArrays() > 0) {
    out->GetPointData()->ShallowCopy(input->GetPointData());
  }

  return out;
}

} // namespace tf::vtk

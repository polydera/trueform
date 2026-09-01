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
#include <trueform/arrangement/mesh/materialize_mesh_triangulation.hpp>
#include <trueform/arrangement/mesh/mesh_triangulation.hpp>
#include <trueform/geometry/triangulation/make_triangulation_faces.hpp>
#include <trueform/vtk/core/make_vtk_cells.hpp>
#include <trueform/vtk/core/make_vtk_points.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/triangulated.hpp>
#include <utility>
#include <vtkPointData.h>
#include <vtkPoints.h>

namespace tf::vtk {

auto triangulated(polydata *input, bool preserve_point_data)
    -> vtkSmartPointer<polydata> {
  if (!input || input->GetNumberOfPolys() == 0) {
    return nullptr;
  }

  const auto polygons = input->polygons();
  const auto triangulation = tf::arrangement::make_mesh_triangulation(polygons);

  auto out = vtkSmartPointer<polydata>::New();
  out->Initialize();

  // THE POINTS ARE THE INPUT'S UNTIL A FACE IS RESOLVED. A face whose loop
  // crosses itself is resolved rather than dropped, and it mints the identity
  // its crossing stands on — a corner past the input's own extent names no
  // point of the input's array, so a build that minted one owns the product's
  // whole table and the input's point data no longer aligns with it. A build
  // that minted none names the input's own ids: there the points are shared
  // and their data carried across.
  if (triangulation.created_points().size() != 0) {
    auto mesh =
        tf::arrangement::materialize_mesh_triangulation(triangulation, polygons);
    out->SetPoints(make_vtk_points(std::move(mesh.points_buffer())));
    out->SetPolys(make_vtk_cells(std::move(mesh.faces_buffer())));
    return out;
  }

  out->SetPoints(input->GetPoints());
  out->SetPolys(
      make_vtk_cells(tf::geometry::make_triangulation_faces(triangulation)));

  if (preserve_point_data && input->GetPointData() &&
      input->GetPointData()->GetNumberOfArrays() > 0) {
    out->GetPointData()->ShallowCopy(input->GetPointData());
  }

  return out;
}

} // namespace tf::vtk

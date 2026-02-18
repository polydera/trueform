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
#include <trueform/remesh/decimated.hpp>
#include <trueform/vtk/core/make_vtk_cells.hpp>
#include <trueform/vtk/core/make_vtk_points.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/decimated.hpp>

namespace tf::vtk {

auto decimated(polydata *input, float target_proportion,
               const tf::decimate_config<float> &config)
    -> vtkSmartPointer<polydata> {
  if (!input || input->GetNumberOfPolys() == 0) {
    return nullptr;
  }

  auto polys = make_polys<3>(input->GetPolys());
  auto polygons = tf::make_polygons(polys, input->points());
  auto [mesh, he] = tf::decimated(polygons, target_proportion, config);

  auto out = vtkSmartPointer<polydata>::New();
  out->SetPoints(make_vtk_points(mesh.points_buffer()));
  out->SetPolys(make_vtk_cells(mesh.faces_buffer()));
  out->set_half_edges(std::move(he));
  return out;
}

} // namespace tf::vtk

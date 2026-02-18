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
#include <trueform/remesh/isotropic_remeshed.hpp>
#include <trueform/vtk/core/make_vtk_cells.hpp>
#include <trueform/vtk/core/make_vtk_points.hpp>
#include <trueform/vtk/core/polydata.hpp>
#include <trueform/vtk/functions/isotropic_remeshed.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

auto isotropic_remeshed(polydata *input,
                        const tf::remesh_config<float> &config)
    -> vtkSmartPointer<polydata> {
  if (!input || input->GetNumberOfPolys() == 0) {
    return nullptr;
  }

  auto polys = make_polys<3>(input->GetPolys());
  auto polygons = tf::make_polygons(polys, input->points());
  auto [mesh, he] = tf::isotropic_remeshed(polygons, config);

  auto out = vtkSmartPointer<polydata>::New();
  out->SetPoints(make_vtk_points(mesh.points_buffer()));
  out->SetPolys(make_vtk_cells(mesh.faces_buffer()));
  out->set_half_edges(std::move(he));
  return out;
}

auto isotropic_remeshed(polydata *input, float target_length)
    -> vtkSmartPointer<polydata> {
  tf::remesh_config<float> config;
  config.target_length = target_length;
  return isotropic_remeshed(input, config);
}

auto isotropic_remeshed(std::pair<polydata *, vtkMatrix4x4 *> input,
                        const tf::remesh_config<float> &config)
    -> vtkSmartPointer<polydata> {
  auto [mesh, matrix] = input;
  if (!mesh || mesh->GetNumberOfPolys() == 0) {
    return nullptr;
  }

  tf::frame<double, 3> frame;
  frame.fill(matrix->GetData());

  auto polys = make_polys<3>(mesh->GetPolys());
  auto polygons = tf::make_polygons(polys, mesh->points()) | tf::tag(frame);
  auto [result, he] = tf::isotropic_remeshed(polygons, config);

  auto out = vtkSmartPointer<polydata>::New();
  out->SetPoints(make_vtk_points(result.points_buffer()));
  out->SetPolys(make_vtk_cells(result.faces_buffer()));
  out->set_half_edges(std::move(he));
  return out;
}

auto isotropic_remeshed(std::pair<polydata *, vtkMatrix4x4 *> input,
                        float target_length)
    -> vtkSmartPointer<polydata> {
  tf::remesh_config<float> config;
  config.target_length = target_length;
  return isotropic_remeshed(input, config);
}

} // namespace tf::vtk

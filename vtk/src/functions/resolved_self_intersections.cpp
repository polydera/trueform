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
#include <trueform/cut/embedded_self_intersection_curves.hpp>
#include <trueform/vtk/core/make_vtk_polydata.hpp>
#include <trueform/vtk/functions/resolved_self_intersections.hpp>

namespace tf::vtk {

namespace {

auto make_base(polydata *in) {
  return in->polygons() | tf::tag(in->face_membership()) |
         tf::tag(in->manifold_edge_link()) | tf::tag(in->poly_tree());
}

} // namespace

auto resolved_self_intersections(polydata *input) -> vtkSmartPointer<polydata> {
  if (!input) {
    return nullptr;
  }

  auto result = tf::embedded_self_intersection_curves(make_base(input));
  auto out = vtkSmartPointer<polydata>::New();
  out->ShallowCopy(make_vtk_polydata(std::move(result)));
  return out;
}

auto resolved_self_intersections(polydata *input, tf::return_curves_t)
    -> std::tuple<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>> {
  if (!input) {
    return {nullptr, nullptr};
  }

  auto [result, curves] =
      tf::embedded_self_intersection_curves(make_base(input), tf::return_curves);

  auto out_mesh = vtkSmartPointer<polydata>::New();
  out_mesh->ShallowCopy(make_vtk_polydata(std::move(result)));

  auto out_curves = vtkSmartPointer<polydata>::New();
  out_curves->ShallowCopy(make_vtk_polydata(std::move(curves)));

  return {out_mesh, out_curves};
}

} // namespace tf::vtk

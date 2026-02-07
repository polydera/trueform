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
#pragma once
#include <trueform/cut/embedded_intersection_curves.hpp>
#include <trueform/vtk/core/make_vtk_polydata.hpp>
#include <trueform/vtk/functions/embedded_intersection_curves.hpp>

namespace tf::vtk::impl {

inline auto make_frame(vtkMatrix4x4 *matrix) -> tf::frame<double, 3> {
  tf::frame<double, 3> frame;
  frame.fill(matrix->GetData());
  return frame;
}

inline auto make_base(polydata *in) {
  return in->polygons() | tf::tag(in->face_membership()) |
         tf::tag(in->manifold_edge_link()) | tf::tag(in->poly_tree());
}

template <typename F0, typename F1>
auto compute_embedded_intersection_curves(F0 &&form0, F1 &&form1)
    -> vtkSmartPointer<polydata> {
  auto mesh = tf::embedded_intersection_curves(form0, form1);

  auto out = vtkSmartPointer<polydata>::New();
  out->ShallowCopy(make_vtk_polydata(std::move(mesh)));

  return out;
}

template <typename F0, typename F1>
auto compute_embedded_intersection_curves_with_curves(F0 &&form0, F1 &&form1)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>> {
  auto [mesh, curves] =
      tf::embedded_intersection_curves(form0, form1, tf::return_curves);

  auto out = vtkSmartPointer<polydata>::New();
  out->ShallowCopy(make_vtk_polydata(std::move(mesh)));

  auto out_curves = vtkSmartPointer<polydata>::New();
  out_curves->ShallowCopy(make_vtk_polydata(std::move(curves)));

  return {out, out_curves};
}

template <typename Runner>
auto dispatch_eic(polydata *in0, polydata *in1, Runner &&runner) {
  return runner(make_base(in0), make_base(in1));
}

} // namespace tf::vtk::impl

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
#include <trueform/cut/make_boolean.hpp>
#include <trueform/vtk/core/make_vtk_array.hpp>
#include <trueform/vtk/core/make_vtk_polydata.hpp>
#include <trueform/vtk/functions/make_boolean.hpp>
#include <vtkCellData.h>
#include <vtkSignedCharArray.h>

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
auto compute_boolean(F0 &&form0, F1 &&form1, tf::boolean_op op)
    -> vtkSmartPointer<polydata> {
  auto [mesh, labels] = tf::make_boolean(form0, form1, op);

  auto out = vtkSmartPointer<polydata>::New();
  out->ShallowCopy(make_vtk_polydata(std::move(mesh)));

  auto label_array = make_vtk_array(std::move(labels));
  label_array->SetName("Labels");
  out->GetCellData()->SetScalars(label_array);

  return out;
}

template <typename F0, typename F1>
auto compute_boolean_with_curves(F0 &&form0, F1 &&form1, tf::boolean_op op)
    -> std::pair<vtkSmartPointer<polydata>, vtkSmartPointer<polydata>> {
  auto [mesh, labels, curves] =
      tf::make_boolean(form0, form1, op, tf::return_curves);

  auto out = vtkSmartPointer<polydata>::New();
  out->ShallowCopy(make_vtk_polydata(std::move(mesh)));

  auto label_array = make_vtk_array(std::move(labels));
  label_array->SetName("Labels");
  out->GetCellData()->SetScalars(label_array);

  auto out_curves = vtkSmartPointer<polydata>::New();
  out_curves->ShallowCopy(make_vtk_polydata(std::move(curves)));

  return {out, out_curves};
}

template <typename Runner>
auto dispatch(polydata *in0, polydata *in1, Runner &&runner) {
  return runner(make_base(in0), make_base(in1));
}

} // namespace tf::vtk::impl

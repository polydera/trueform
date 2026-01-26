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
#include <trueform/spatial.hpp>
#include <trueform/vtk/functions/intersects.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

auto intersects(polydata *input0, polydata *input1) -> bool {
  auto form0 = input0->polygons() | tf::tag(input0->poly_tree());
  auto form1 = input1->polygons() | tf::tag(input1->poly_tree());
  return tf::intersects(form0, form1);
}

auto intersects(std::pair<polydata *, vtkMatrix4x4 *> input0, polydata *input1)
    -> bool {
  auto [mesh0, matrix0] = input0;
  tf::frame<double, 3> frame0;
  frame0.fill(matrix0->GetData());
  auto form0 = mesh0->polygons() | tf::tag(mesh0->poly_tree()) | tf::tag(frame0);
  auto form1 = input1->polygons() | tf::tag(input1->poly_tree());
  return tf::intersects(form0, form1);
}

auto intersects(polydata *input0, std::pair<polydata *, vtkMatrix4x4 *> input1)
    -> bool {
  auto [mesh1, matrix1] = input1;
  tf::frame<double, 3> frame1;
  frame1.fill(matrix1->GetData());
  auto form0 = input0->polygons() | tf::tag(input0->poly_tree());
  auto form1 = mesh1->polygons() | tf::tag(mesh1->poly_tree()) | tf::tag(frame1);
  return tf::intersects(form0, form1);
}

auto intersects(std::pair<polydata *, vtkMatrix4x4 *> input0,
                std::pair<polydata *, vtkMatrix4x4 *> input1) -> bool {
  auto [mesh0, matrix0] = input0;
  auto [mesh1, matrix1] = input1;
  tf::frame<double, 3> frame0;
  frame0.fill(matrix0->GetData());
  tf::frame<double, 3> frame1;
  frame1.fill(matrix1->GetData());
  auto form0 = mesh0->polygons() | tf::tag(mesh0->poly_tree()) | tf::tag(frame0);
  auto form1 = mesh1->polygons() | tf::tag(mesh1->poly_tree()) | tf::tag(frame1);
  return tf::intersects(form0, form1);
}

} // namespace tf::vtk

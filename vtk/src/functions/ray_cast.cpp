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
#include <trueform/spatial/ray_cast.hpp>
#include <trueform/spatial/policy.hpp>
#include <trueform/vtk/functions/ray_cast.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

auto ray_cast(tf::ray<float, 3> ray, polydata *input) -> ray_cast_result {
  return ray_cast(ray, input, tf::ray_config<float>{});
}

auto ray_cast(tf::ray<float, 3> ray, polydata *input,
              tf::ray_config<float> config) -> ray_cast_result {
  auto form = input->polygons() | tf::tag(input->poly_tree());
  return tf::ray_cast(ray, form, config);
}

auto ray_cast(tf::ray<float, 3> ray, std::pair<polydata *, vtkMatrix4x4 *> input)
    -> ray_cast_result {
  return ray_cast(ray, input, tf::ray_config<float>{});
}

auto ray_cast(tf::ray<float, 3> ray, std::pair<polydata *, vtkMatrix4x4 *> input,
              tf::ray_config<float> config) -> ray_cast_result {
  auto [mesh, matrix] = input;
  tf::frame<double, 3> frame;
  frame.fill(matrix->GetData());
  auto form = mesh->polygons() | tf::tag(mesh->poly_tree()) | tf::tag(frame);
  return tf::ray_cast(ray, form, config);
}

} // namespace tf::vtk

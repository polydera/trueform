/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/spatial/ray_hit.hpp>
#include <trueform/vtk/functions/ray_hit.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

auto ray_hit(tf::ray<float, 3> ray, polydata *input) -> ray_hit_result {
  return ray_hit(ray, input, tf::ray_config<float>{});
}

auto ray_hit(tf::ray<float, 3> ray, polydata *input,
             tf::ray_config<float> config) -> ray_hit_result {
  auto form = tf::make_form(input->poly_tree(), input->polygons());
  return tf::ray_hit(ray, form, config);
}

auto ray_hit(tf::ray<float, 3> ray, std::pair<polydata *, vtkMatrix4x4 *> input)
    -> ray_hit_result {
  return ray_hit(ray, input, tf::ray_config<float>{});
}

auto ray_hit(tf::ray<float, 3> ray, std::pair<polydata *, vtkMatrix4x4 *> input,
             tf::ray_config<float> config) -> ray_hit_result {
  auto [mesh, matrix] = input;
  tf::frame<double, 3> frame;
  frame.fill(matrix->GetData());
  auto form = tf::make_form(frame, mesh->poly_tree(), mesh->polygons());
  return tf::ray_hit(ray, form, config);
}

} // namespace tf::vtk

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <trueform/spatial/ray_cast.hpp>
#include <trueform/vtk/functions/ray_cast.hpp>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

auto ray_cast(tf::ray<float, 3> ray, polydata *input)
    -> std::optional<ray_cast_result> {
  return ray_cast(ray, input, tf::ray_config<float>{});
}

auto ray_cast(tf::ray<float, 3> ray, polydata *input,
              tf::ray_config<float> config) -> std::optional<ray_cast_result> {
  auto form = tf::make_form(input->poly_tree(), input->polygons());
  auto hit = tf::ray_cast(ray, form, config);

  if (!hit)
    return std::nullopt;

  return ray_cast_result{hit.element, hit.info.t};
}

auto ray_cast(tf::ray<float, 3> ray,
              std::pair<polydata *, vtkMatrix4x4 *> input)
    -> std::optional<ray_cast_result> {
  return ray_cast(ray, input, tf::ray_config<float>{});
}

auto ray_cast(tf::ray<float, 3> ray,
              std::pair<polydata *, vtkMatrix4x4 *> input,
              tf::ray_config<float> config) -> std::optional<ray_cast_result> {
  auto [mesh, matrix] = input;
  tf::frame<double, 3> frame;
  frame.fill(matrix->GetData());
  auto form = tf::make_form(frame, mesh->poly_tree(), mesh->polygons());
  auto hit = tf::ray_cast(ray, form, config);

  if (!hit)
    return std::nullopt;

  return ray_cast_result{hit.element, hit.info.t};
}

} // namespace tf::vtk

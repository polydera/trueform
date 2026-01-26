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
#include <trueform/vtk/core/make_world_ray.hpp>
#include <vtkCamera.h>

namespace tf::vtk {

auto make_world_ray(vtkRenderer *renderer, int x, int y) -> tf::ray<float, 3> {
  tf::ray<float, 3> ray{};

  auto *camera = renderer->GetActiveCamera();
  if (!camera)
    return ray;

  if (camera->GetParallelProjection()) {
    renderer->SetDisplayPoint(x, y, 0.0);
    renderer->DisplayToWorld();
    auto *world_point = renderer->GetWorldPoint();
    ray.origin[0] = static_cast<float>(world_point[0]);
    ray.origin[1] = static_cast<float>(world_point[1]);
    ray.origin[2] = static_cast<float>(world_point[2]);

    auto *dir = camera->GetDirectionOfProjection();
    ray.direction[0] = static_cast<float>(dir[0]);
    ray.direction[1] = static_cast<float>(dir[1]);
    ray.direction[2] = static_cast<float>(dir[2]);
  } else {
    auto *focal = camera->GetFocalPoint();
    renderer->SetWorldPoint(focal[0], focal[1], focal[2], 1.0);
    renderer->WorldToDisplay();

    auto *display = renderer->GetDisplayPoint();
    renderer->SetDisplayPoint(x, y, display[2]);
    renderer->DisplayToWorld();

    auto *world_point = renderer->GetWorldPoint();
    auto w = world_point[3];
    ray.origin[0] = static_cast<float>(world_point[0] / w);
    ray.origin[1] = static_cast<float>(world_point[1] / w);
    ray.origin[2] = static_cast<float>(world_point[2] / w);

    auto *camera_pos = camera->GetPosition();
    ray.direction[0] = ray.origin[0] - static_cast<float>(camera_pos[0]);
    ray.direction[1] = ray.origin[1] - static_cast<float>(camera_pos[1]);
    ray.direction[2] = ray.origin[2] - static_cast<float>(camera_pos[2]);
    tf::normalize(ray.direction);

    ray.origin[0] = static_cast<float>(camera_pos[0]);
    ray.origin[1] = static_cast<float>(camera_pos[1]);
    ray.origin[2] = static_cast<float>(camera_pos[2]);
  }

  return ray;
}

} // namespace tf::vtk

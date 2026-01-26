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
#include <trueform/vtk/functions/pick.hpp>
#include <vtkMapper.h>
#include <vtkMatrix4x4.h>

namespace tf::vtk {

namespace {

template <typename Actors>
auto pick_impl(tf::ray<float, 3> ray, Actors &actors) -> pick_result {
  tf::tree_ray_info<vtkIdType, tf::ray_cast_info<float>> best_hit;
  tf::ray_config<float> config{};
  pick_result result;

  for (auto &actor_ref : actors) {
    vtkActor *actor = nullptr;
    if constexpr (std::is_pointer_v<std::decay_t<decltype(actor_ref)>>) {
      actor = actor_ref;
    } else {
      actor = actor_ref.Get();
    }

    if (!actor || !actor->GetVisibility())
      continue;

    auto *mapper = actor->GetMapper();
    if (!mapper)
      continue;

    auto *input = polydata::SafeDownCast(mapper->GetInput());
    if (!input)
      continue;

    auto *matrix = actor->GetMatrix();
    tf::tree_ray_info<vtkIdType, tf::ray_cast_info<float>> hit;

    if (matrix) {
      tf::frame<float, 3> frame;
      frame.fill(matrix->GetData());
      auto form = input->polygons() | tf::tag(input->poly_tree()) | tf::tag(frame);
      hit = tf::ray_cast(ray, form, config);
    } else {
      auto form = input->polygons() | tf::tag(input->poly_tree());
      hit = tf::ray_cast(ray, form, config);
    }

    if (hit) {
      best_hit = hit;
      config.max_t = hit.info.t;
      result.actor = actor;
    }
  }

  if (result) {
    result.cell_id = best_hit.element;
    result.position = ray.origin + best_hit.info.t * ray.direction;
    result.t = best_hit.info.t;
  }
  return result;
}

} // namespace

auto pick(tf::ray<float, 3> ray, std::vector<vtkActor *> &actors)
    -> pick_result {
  return pick_impl(ray, actors);
}

auto pick(tf::ray<float, 3> ray,
          tf::range<vtkActor **, tf::dynamic_size> actors)
    -> pick_result {
  return pick_impl(ray, actors);
}

auto pick(tf::ray<float, 3> ray, std::vector<vtkSmartPointer<vtkActor>> &actors)
    -> pick_result {
  return pick_impl(ray, actors);
}

auto pick(tf::ray<float, 3> ray,
          tf::range<vtkSmartPointer<vtkActor> *, tf::dynamic_size> actors)
    -> pick_result {
  return pick_impl(ray, actors);
}

} // namespace tf::vtk

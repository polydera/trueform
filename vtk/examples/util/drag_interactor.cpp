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
#include "drag_interactor.hpp"
#include <vtkCamera.h>
#include <vtkMatrix4x4.h>
#include <vtkObjectFactory.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

namespace tf::vtk::examples {

vtkStandardNewMacro(drag_interactor);

drag_interactor::drag_interactor() = default;

auto drag_interactor::add_actor(vtkActor *actor, vtkRenderer *renderer) -> void {
  _actors[renderer].push_back(actor);
}

auto drag_interactor::set_callback(callback_t cb) -> void {
  _callback = std::move(cb);
}

auto drag_interactor::actors_for_renderer(vtkRenderer *renderer)
    -> std::vector<vtkActor *> & {
  return _actors[renderer];
}

auto drag_interactor::OnLeftButtonDown() -> void {
  int x = this->Interactor->GetEventPosition()[0];
  int y = this->Interactor->GetEventPosition()[1];
  auto *renderer = this->Interactor->FindPokedRenderer(x, y);
  if (!renderer)
    return;

  auto &actors = actors_for_renderer(renderer);
  auto ray = make_world_ray(renderer, x, y);
  auto hit = pick(ray, actors);

  if (hit) {
    _dragging_actor = hit.actor;
    _dragging_renderer = renderer;
    _last_point = hit.position;

    // Create drag plane perpendicular to camera
    auto *camera = renderer->GetActiveCamera();
    auto *focal = camera->GetFocalPoint();
    auto *pos = camera->GetPosition();
    tf::vector<float, 3> normal{static_cast<float>(focal[0] - pos[0]),
                                static_cast<float>(focal[1] - pos[1]),
                                static_cast<float>(focal[2] - pos[2])};
    _drag_plane = tf::make_plane(tf::normalized(normal), _last_point);
  } else {
    vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
  }
}

auto drag_interactor::OnLeftButtonUp() -> void {
  if (_dragging_actor) {
    _dragging_actor = nullptr;
    _dragging_renderer = nullptr;
  } else {
    vtkInteractorStyleTrackballCamera::OnLeftButtonUp();
  }
}

auto drag_interactor::OnMouseMove() -> void {
  if (_dragging_actor) {
    int x = this->Interactor->GetEventPosition()[0];
    int y = this->Interactor->GetEventPosition()[1];

    // Use the renderer where drag started, not the current poked renderer
    if (!_dragging_renderer)
      return;

    auto ray = make_world_ray(_dragging_renderer, x, y);
    auto hit = tf::ray_hit(ray, _drag_plane);

    tf::vector<float, 3> delta = hit.point - _last_point;
    _last_point = hit.point;

    // Update actor matrix
    auto *matrix = _dragging_actor->GetUserMatrix();
    if (!matrix) {
      auto m = vtkSmartPointer<vtkMatrix4x4>::New();
      m->Identity();
      _dragging_actor->SetUserMatrix(m);
      matrix = m;
    }
    matrix->SetElement(0, 3, matrix->GetElement(0, 3) + delta[0]);
    matrix->SetElement(1, 3, matrix->GetElement(1, 3) + delta[1]);
    matrix->SetElement(2, 3, matrix->GetElement(2, 3) + delta[2]);
    matrix->Modified();

    if (_callback) {
      auto &actors = actors_for_renderer(_dragging_renderer);
      _callback(_dragging_actor, actors);
    }

    this->Interactor->Render();
  } else {
    vtkInteractorStyleTrackballCamera::OnMouseMove();
  }
}

} // namespace tf::vtk::examples

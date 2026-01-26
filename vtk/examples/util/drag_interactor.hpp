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
#include <trueform/vtk/core.hpp>
#include <trueform/vtk/functions.hpp>
#include <trueform/core/hash_map.hpp>
#include <vtkInteractorStyleTrackballCamera.h>
#include <functional>
#include <vector>

namespace tf::vtk::examples {

class drag_interactor : public vtkInteractorStyleTrackballCamera {
public:
  using callback_t =
      std::function<void(vtkActor *selected, std::vector<vtkActor *> &all)>;

  static auto New() -> drag_interactor *;
  vtkTypeMacro(drag_interactor, vtkInteractorStyleTrackballCamera);

  auto add_actor(vtkActor *actor, vtkRenderer *renderer) -> void;
  auto set_callback(callback_t cb) -> void;

  auto OnMouseMove() -> void override;
  auto OnLeftButtonDown() -> void override;
  auto OnLeftButtonUp() -> void override;

protected:
  drag_interactor();
  ~drag_interactor() override = default;

private:
  auto actors_for_renderer(vtkRenderer *renderer) -> std::vector<vtkActor *> &;

  tf::hash_map<vtkRenderer *, std::vector<vtkActor *>> _actors;
  vtkActor *_dragging_actor = nullptr;
  vtkRenderer *_dragging_renderer = nullptr;
  tf::plane<float, 3> _drag_plane;
  tf::point<float, 3> _last_point;
  callback_t _callback;
};

} // namespace tf::vtk::examples

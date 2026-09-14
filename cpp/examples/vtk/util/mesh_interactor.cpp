#include "mesh_interactor.hpp"

#include <vtkCamera.h>
#include <vtkObjectFactory.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace trueform::vtk_examples {

vtkStandardNewMacro(mesh_interactor);

auto mesh_interactor::add_mesh(mesh_actor_data &data) -> void {
  _items.push_back({&data});
  data.actor->GetProperty()->SetColor(_normal_color.data());
}

auto mesh_interactor::clear_meshes() -> void {
  _items.clear();
  _hovered = nullptr;
  _selected = nullptr;
  _dragging = false;
}

auto mesh_interactor::set_interaction_renderer(vtkRenderer *renderer) -> void {
  _interaction_renderer = renderer;
}

auto mesh_interactor::set_colors(point3<double> normal,
                                 point3<double> highlight) -> void {
  _normal_color = normal;
  _highlight_color = highlight;
  reset_colors();
}

auto mesh_interactor::set_hover_callback(hover_callback callback) -> void {
  _hover_callback = std::move(callback);
}

auto mesh_interactor::set_selected_callback(mesh_callback callback) -> void {
  _selected_callback = std::move(callback);
}

auto mesh_interactor::set_dragged_callback(drag_callback callback) -> void {
  _dragged_callback = std::move(callback);
}

auto mesh_interactor::set_released_callback(hover_callback callback) -> void {
  _released_callback = std::move(callback);
}

auto mesh_interactor::selected_mesh() const -> mesh_actor_data * {
  return _selected;
}

auto mesh_interactor::camera_ray(vtkRenderer *renderer, int x, int y)
    -> tf::cpp::primitive<float> {
  auto *camera = renderer->GetActiveCamera();
  const auto *focal = camera->GetFocalPoint();
  renderer->SetWorldPoint(focal[0], focal[1], focal[2], 1.0);
  renderer->WorldToDisplay();
  const auto depth = renderer->GetDisplayPoint()[2];
  renderer->SetDisplayPoint(static_cast<double>(x), static_cast<double>(y),
                            depth);
  renderer->DisplayToWorld();
  const auto *world = renderer->GetWorldPoint();
  const auto inverse_w = world[3] == 0.0 ? 1.0 : 1.0 / world[3];
  const auto *camera_position = camera->GetPosition();
  const point3<float> origin{static_cast<float>(camera_position[0]),
                             static_cast<float>(camera_position[1]),
                             static_cast<float>(camera_position[2])};
  const point3<float> through{static_cast<float>(world[0] * inverse_w),
                              static_cast<float>(world[1] * inverse_w),
                              static_cast<float>(world[2] * inverse_w)};
  return tf::cpp::primitive<float>(
      tf::cpp::primitive_kind::ray,
      make_array<float>({origin[0], origin[1], origin[2],
                         through[0] - origin[0], through[1] - origin[1],
                         through[2] - origin[2]},
                        {2, 3}));
}

auto mesh_interactor::closest_hit(const tf::cpp::primitive<float> &ray)
    -> std::optional<hit> {
  std::optional<hit> closest;
  tf::cpp::ray_cast_options<float> options;
  options.max_t = std::numeric_limits<float>::max();
  for (const auto &entry : _items) {
    auto result = tf::cpp::ray_cast(ray, entry.data->mesh(), options).scalar();
    if (!result.hit)
      continue;
    const auto coordinates = ray.data();
    const point3<float> position{coordinates[0] + result.t * coordinates[3],
                                 coordinates[1] + result.t * coordinates[4],
                                 coordinates[2] + result.t * coordinates[5]};
    closest = hit{entry.data, result.t, position};
    options.max_t = result.t;
  }
  return closest;
}

auto mesh_interactor::make_drag_plane(vtkRenderer *renderer,
                                      const point3<float> &origin) -> void {
  const auto *camera_position = renderer->GetActiveCamera()->GetPosition();
  const auto *focal = renderer->GetActiveCamera()->GetFocalPoint();
  _plane = {static_cast<float>(focal[0] - camera_position[0]),
            static_cast<float>(focal[1] - camera_position[1]),
            static_cast<float>(focal[2] - camera_position[2])};
  const auto length = std::sqrt(_plane[0] * _plane[0] + _plane[1] * _plane[1] +
                                _plane[2] * _plane[2]);
  if (!(length > 0.0F)) {
    _has_plane = false;
    return;
  }
  for (auto &coordinate : _plane)
    coordinate /= length;
  _plane_d =
      -(_plane[0] * origin[0] + _plane[1] * origin[1] + _plane[2] * origin[2]);
  _has_plane = true;
}

auto mesh_interactor::intersect_drag_plane(const tf::cpp::primitive<float> &ray)
    -> std::optional<point3<float>> {
  if (!_has_plane)
    return std::nullopt;
  const auto plane = tf::cpp::primitive<float>(
      tf::cpp::primitive_kind::plane,
      make_array<float>({_plane[0], _plane[1], _plane[2], _plane_d}, {4}));
  const auto result = tf::cpp::ray_cast(ray, plane).scalar();
  if (!result.hit)
    return std::nullopt;
  const auto coordinates = ray.data();
  return point3<float>{coordinates[0] + result.t * coordinates[3],
                       coordinates[1] + result.t * coordinates[4],
                       coordinates[2] + result.t * coordinates[5]};
}

auto mesh_interactor::reset_colors() -> void {
  for (const auto &entry : _items)
    entry.data->actor->GetProperty()->SetColor(_normal_color.data());
}

auto mesh_interactor::refresh_hover(vtkRenderer *renderer, int x, int y)
    -> void {
  if (!renderer ||
      (_interaction_renderer && renderer != _interaction_renderer)) {
    if (_hovered) {
      reset_colors();
      _hovered = nullptr;
      if (_hover_callback)
        _hover_callback(nullptr);
    }
    return;
  }
  const auto candidate = closest_hit(camera_ray(renderer, x, y));
  auto *next = candidate ? candidate->data : nullptr;
  if (next != _hovered) {
    reset_colors();
    _hovered = next;
    if (_hovered)
      _hovered->actor->GetProperty()->SetColor(_highlight_color.data());
    if (_hover_callback)
      _hover_callback(_hovered);
  }
  if (candidate) {
    _last_point = candidate->position;
    make_drag_plane(renderer, candidate->position);
  }
}

auto mesh_interactor::OnMouseMove() -> void {
  const auto *position = this->Interactor->GetEventPosition();
  if (_dragging && _selected && _drag_renderer) {
    const auto ray = camera_ray(_drag_renderer, position[0], position[1]);
    const auto next = intersect_drag_plane(ray);
    if (!next)
      return;
    const point3<float> delta{(*next)[0] - _last_point[0],
                              (*next)[1] - _last_point[1],
                              (*next)[2] - _last_point[2]};
    _last_point = *next;
    auto transform = from_vtk_matrix<float>(_selected->matrix);
    transform[3] += delta[0];
    transform[7] += delta[1];
    transform[11] += delta[2];
    synchronize(*_selected, transform);
    if (_dragged_callback)
      _dragged_callback(*_selected, delta);
    this->Interactor->Render();
    return;
  }
  if (_camera_mode) {
    vtkInteractorStyleTrackballCamera::OnMouseMove();
    return;
  }
  auto *renderer =
      this->Interactor->FindPokedRenderer(position[0], position[1]);
  refresh_hover(renderer, position[0], position[1]);
  this->Interactor->Render();
}

auto mesh_interactor::OnLeftButtonDown() -> void {
  const auto *position = this->Interactor->GetEventPosition();
  auto *renderer =
      this->Interactor->FindPokedRenderer(position[0], position[1]);
  refresh_hover(renderer, position[0], position[1]);
  if (_hovered) {
    _selected = _hovered;
    _drag_renderer = renderer;
    _dragging = true;
    this->Interactor->GetRenderWindow()->HideCursor();
    if (_selected_callback)
      _selected_callback(*_selected);
    return;
  }
  _camera_mode = true;
  vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
}

auto mesh_interactor::OnLeftButtonUp() -> void {
  if (_dragging) {
    _dragging = false;
    this->Interactor->GetRenderWindow()->ShowCursor();
    const auto *position = this->Interactor->GetEventPosition();
    auto *renderer =
        this->Interactor->FindPokedRenderer(position[0], position[1]);
    reset_colors();
    _hovered = nullptr;
    refresh_hover(renderer, position[0], position[1]);
    _selected = nullptr;
    _drag_renderer = nullptr;
    if (_released_callback)
      _released_callback(_hovered);
    return;
  }
  if (_camera_mode) {
    _camera_mode = false;
    vtkInteractorStyleTrackballCamera::OnLeftButtonUp();
  }
}

} // namespace trueform::vtk_examples

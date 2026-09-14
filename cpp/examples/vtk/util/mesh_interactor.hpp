#pragma once

#include "vtk_examples.hpp"

#include <trueform/cpp/spatial/primitive.hpp>
#include <trueform/cpp/spatial/ray_cast.hpp>

#include <vtkInteractorStyleTrackballCamera.h>

#include <array>
#include <functional>
#include <optional>
#include <vector>

class vtkRenderer;

namespace trueform::vtk_examples {

class mesh_interactor : public vtkInteractorStyleTrackballCamera {
public:
  struct item {
    mesh_actor_data *data = nullptr;
  };
  using mesh_callback = std::function<void(mesh_actor_data &)>;
  using drag_callback =
      std::function<void(mesh_actor_data &, const point3<float> &)>;
  using hover_callback = std::function<void(mesh_actor_data *)>;

  static auto New() -> mesh_interactor *;
  vtkTypeMacro(mesh_interactor, vtkInteractorStyleTrackballCamera);

  auto add_mesh(mesh_actor_data &data) -> void;
  auto clear_meshes() -> void;
  auto set_interaction_renderer(vtkRenderer *renderer) -> void;
  auto set_colors(point3<double> normal, point3<double> highlight) -> void;
  auto set_hover_callback(hover_callback callback) -> void;
  auto set_selected_callback(mesh_callback callback) -> void;
  auto set_dragged_callback(drag_callback callback) -> void;
  auto set_released_callback(hover_callback callback) -> void;
  auto selected_mesh() const -> mesh_actor_data *;

  auto OnMouseMove() -> void override;
  auto OnLeftButtonDown() -> void override;
  auto OnLeftButtonUp() -> void override;

protected:
  mesh_interactor() = default;
  ~mesh_interactor() override = default;

  auto camera_ray(vtkRenderer *renderer, int x, int y)
      -> tf::cpp::primitive<float>;
  auto refresh_hover(vtkRenderer *renderer, int x, int y) -> void;

private:
  struct hit {
    mesh_actor_data *data = nullptr;
    float t = 0.0F;
    point3<float> position{};
  };

  auto closest_hit(const tf::cpp::primitive<float> &ray) -> std::optional<hit>;
  auto make_drag_plane(vtkRenderer *renderer, const point3<float> &origin)
      -> void;
  auto intersect_drag_plane(const tf::cpp::primitive<float> &ray)
      -> std::optional<point3<float>>;
  auto reset_colors() -> void;

  std::vector<item> _items;
  vtkRenderer *_interaction_renderer = nullptr;
  mesh_actor_data *_hovered = nullptr;
  mesh_actor_data *_selected = nullptr;
  vtkRenderer *_drag_renderer = nullptr;
  point3<float> _last_point{};
  point3<float> _plane{};
  float _plane_d = 0.0F;
  bool _has_plane = false;
  bool _dragging = false;
  bool _camera_mode = false;
  point3<double> _normal_color{0.8, 0.8, 0.8};
  point3<double> _highlight_color{0.85, 0.85, 0.9};
  hover_callback _hover_callback;
  mesh_callback _selected_callback;
  drag_callback _dragged_callback;
  hover_callback _released_callback;
};

} // namespace trueform::vtk_examples

#pragma once
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include "utils/bridge_web.h"

class tf_bridge_interface {
public:
  virtual ~tf_bridge_interface() = default;

  // Add new mesh data and return its index
  auto add_mesh_data(tf::polygons_buffer<int, float, 3, 3> polygons,
                     bool build_topology = true) -> std::size_t {
    mesh_data_store.emplace_back();
    auto &data = mesh_data_store.back();
    data.polygons = std::move(polygons);
    data.tree.build(data.polygons.polygons(), tf::config_tree(4, 4));
    if (build_topology) {
      data.face_membership.emplace();
      data.face_membership->build(data.polygons.polygons());
      data.manifold_edge_link.emplace();
      data.manifold_edge_link->build(data.polygons.faces(),
                                     *data.face_membership);
    }
    return mesh_data_store.size() - 1;
  }

  // Create instance from mesh data
  auto add_instance(std::size_t mesh_data_id) -> std::size_t {
    instances.emplace_back();
    instances.back().mesh_data_id = mesh_data_id;
    return instances.size() - 1;
  }

  // Create instance with custom color
  auto add_instance(std::size_t mesh_data_id, double r, double g, double b)
      -> std::size_t {
    auto id = add_instance(mesh_data_id);
    instances[id].set_color(r, g, b);
    return id;
  }

  auto get_mesh_data_store() -> std::vector<mesh_data> & {
    return mesh_data_store;
  }

  auto get_mesh_data_store() const -> const std::vector<mesh_data> & {
    return mesh_data_store;
  }

  auto get_instances() -> std::vector<instance> & { return instances; }

  auto get_instances() const -> const std::vector<instance> & {
    return instances;
  }

  auto get_mesh_data(std::size_t id) -> mesh_data & {
    return mesh_data_store[id];
  }

  auto get_mesh_data(std::size_t id) const -> const mesh_data & {
    return mesh_data_store[id];
  }

  auto get_instance(std::size_t id) -> instance & { return instances[id]; }

  auto get_instance(std::size_t id) const -> const instance & {
    return instances[id];
  }

  auto ray_hit(tf::ray<float, 3> ray)
      -> std::pair<std::optional<std::size_t>, tf::point<float, 3>> {
    tf::tree_ray_info<int, tf::ray_cast_info<float>> result;
    tf::ray_config<float> config{};
    std::optional<std::size_t> picked_instance;

    for (std::size_t i = 0; i < instances.size(); ++i) {
      auto &inst = instances[i];
      auto &data = mesh_data_store[inst.mesh_data_id];
      auto form = data.polygons.polygons() | tf::tag(data.tree) | tf::tag(inst.frame);
      auto res = tf::ray_cast(ray, form, config);
      if (res) {
        result = res;
        config.max_t = result.info.t;
        picked_instance = i;
      }
    }
    return std::make_pair(picked_instance,
                          ray.origin + result.info.t * ray.direction);
  }

  auto update_frame(std::size_t instance_id) -> void {
    instances[instance_id].update_frame();
  }

protected:
  std::vector<mesh_data> mesh_data_store;
  std::vector<instance> instances;
};

class cursor_interactor_interface {
public:
  explicit cursor_interactor_interface(std::unique_ptr<tf_bridge_interface> in)
      : bridge(std::move(in)) {}
  virtual ~cursor_interactor_interface() = default;

  result_mesh result;
  result_mesh curves;

protected:
  std::unique_ptr<tf_bridge_interface> bridge;
  int time_index = 0;

  tf::plane<float, 3> moving_plane;
  tf::point<float, 3> last_point;
  tf::vector<float, 3> dx;
  std::optional<std::size_t> selected_instance;
  bool selected_mode = false;
  bool camera_mode = false;

  auto add_time(std::vector<float> &times, float t) {
    if (times.size() < 10) {
      times.push_back(t);
    } else {
      times[time_index] = t;
    }
    time_index = (time_index + 1) % 10;
    float sum = 0;
    for (auto time : times) {
      sum += time;
    }
    auto avg_time = sum / times.size();
    m_time = avg_time;
    return avg_time;
  }

  auto make_moving_plane(tf::point<float, 3> origin,
                         std::array<float, 3> camera_position,
                         std::array<float, 3> camera_focal_point) -> void {
    auto normal =
        tf::make_unit_vector(tf::make_vector_view<3>(camera_focal_point.data()) -
                             tf::make_vector_view<3>(camera_position.data()));
    moving_plane = tf::make_plane(normal, origin);
  }

  auto move_selected(std::size_t instance_id) -> void {
    auto &inst = bridge->get_instance(instance_id);
    for (int i = 0; i < 3; ++i) {
      inst.matrix[i * 4 + 3] += dx[i];
    }
    bridge->update_frame(instance_id);
  }

public:
  float m_time = 0.f;
  float m_pick_time = 0.f;
  std::size_t total_polygons = 0;

  auto get_mesh_data_store() -> std::vector<mesh_data> & {
    return bridge->get_mesh_data_store();
  }

  auto get_instances() -> std::vector<instance> & {
    return bridge->get_instances();
  }

  auto add_mesh_data(tf::polygons_buffer<int, float, 3, 3> polygons,
                     bool build_topology = true) -> std::size_t {
    return bridge->add_mesh_data(std::move(polygons), build_topology);
  }

  auto add_instance(std::size_t mesh_data_id) -> std::size_t {
    return bridge->add_instance(mesh_data_id);
  }

  auto add_instance(std::size_t mesh_data_id, double r, double g, double b)
      -> std::size_t {
    return bridge->add_instance(mesh_data_id, r, g, b);
  }

  virtual auto OnLeftButtonDown() -> bool {
    if (selected_instance) {
      selected_mode = true;
      return true;
    }
    camera_mode = true;
    return false;
  }

  virtual auto OnLeftButtonUp() -> bool {
    if (selected_mode) {
      selected_mode = false;
      return true;
    }
    if (camera_mode) {
      camera_mode = false;
    }
    return false;
  }

  virtual auto OnMouseMove(std::array<float, 3> origin,
                           std::array<float, 3> direction,
                           std::array<float, 3> camera_position,
                           std::array<float, 3> camera_focal_point) -> bool {
    tf::ray<float, 3> ray{origin, direction};
    if (!selected_mode && !camera_mode) {
      auto [instance_id, point] = bridge->ray_hit(ray);
      if (instance_id) {
        make_moving_plane(point, camera_position, camera_focal_point);
        last_point = point;
      }
      selected_instance = instance_id;
      return true;
    }
    if (selected_mode && selected_instance) {
      auto next_point = tf::ray_hit(ray, moving_plane).point;
      dx = next_point - last_point;
      last_point = next_point;
      move_selected(*selected_instance);
      return true;
    }
    return false;
  }

  virtual auto OnKeyPress(std::string) -> bool { return false; }

  virtual auto OnMouseWheel(int, bool) -> bool { return false; }
};

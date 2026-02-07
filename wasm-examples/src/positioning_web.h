#pragma once
#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "trueform/io/read_stl.hpp"
#include "trueform/random.hpp"
#include "trueform/spatial.hpp"
#include "main.h"
#include "utils/bridge_web.h"
#include "utils/cursor_interactor_interface.h"
#include "utils/utils.h"

class positioning_bridge : public tf_bridge_interface {
private:
  auto other_index(std::size_t id0) const -> std::size_t {
    if (instances.size() < 2) {
      return id0;
    }
    return (id0 + 1) % instances.size();
  }

public:
  auto closest_metric_point_pair(std::size_t instance_id0) {
    if (instances.size() < 2) {
      throw std::runtime_error("Need at least two meshes for positioning.");
    }
    std::size_t id1 = other_index(instance_id0);

    auto &inst0 = instances[instance_id0];
    auto &inst1 = instances[id1];
    auto &data0 = mesh_data_store[inst0.mesh_data_id];
    auto &data1 = mesh_data_store[inst1.mesh_data_id];

    auto form0 = data0.polygons.polygons() | tf::tag(data0.tree) | tf::tag(inst0.frame);
    auto form1 = data1.polygons.polygons() | tf::tag(data1.tree) | tf::tag(inst1.frame);
    return tf::neighbor_search(form0, form1);
  }

  auto intersects_other(std::size_t instance_id0) -> bool {
    if (instances.size() < 2) {
      return false;
    }
    std::size_t id1 = other_index(instance_id0);

    auto &inst0 = instances[instance_id0];
    auto &inst1 = instances[id1];
    auto &data0 = mesh_data_store[inst0.mesh_data_id];
    auto &data1 = mesh_data_store[inst1.mesh_data_id];

    auto form0 = data0.polygons.polygons() | tf::tag(data0.tree) | tf::tag(inst0.frame);
    auto form1 = data1.polygons.polygons() | tf::tag(data1.tree) | tf::tag(inst1.frame);
    return tf::intersects(form0, form1);
  }
};

class cursor_interactor_positioning : public cursor_interactor_interface {
public:
  cursor_interactor_positioning()
      : cursor_interactor_interface(std::make_unique<positioning_bridge>()) {}

  auto reset_active_color(std::size_t instance_id) -> void {
    bridge->get_instance(instance_id)
        .set_color(normal_mesh_color[0], normal_mesh_color[1],
                   normal_mesh_color[2]);
  }

private:
  tf::point<double, 3> normal_mesh_color{0.8, 0.8, 0.8};
  std::vector<float> positioning_times;

  bool moving_mode = false;
  tf::point<double, 3> m_pt1;
  tf::point<double, 3> m_prev_pt;
  tf::ray<float, 3> m_ray;
  tf::ray<float, 3> m_focal_ray;

  // Closest points visualization
  tf::point<double, 3> m_closest_pt0;
  tf::point<double, 3> m_closest_pt1;
  bool m_has_closest_points = false;
  float m_aabb_diagonal = 1.0f;

  auto add_position_time(float t) -> void {
    auto avg = add_time(positioning_times, t);
    m_time = avg;
  }

  auto move_instance(std::size_t instance_id, const tf::vector<float, 3> &delta)
      -> void {
    auto &inst = bridge->get_instance(instance_id);
    for (int i = 0; i < 3; ++i) {
      inst.matrix[i * 4 + 3] += delta[i];
    }
    bridge->update_frame(instance_id);
  }

  auto position_them(std::array<double, 3> focal_point,
                     emscripten::val lambda_set_focal, float dt) -> float {
    if (auto *pos_bridge = static_cast<positioning_bridge *>(bridge.get())) {
      if (!selected_instance) {
        return 2.0f;
      }
      auto eps = 0.01f;
      if (dt == 0) {
        if (pos_bridge->intersects_other(*selected_instance)) {
          m_has_closest_points = false;
          return 1.0f;
        }
        auto neighbors = pos_bridge->closest_metric_point_pair(*selected_instance);
        auto pt0 = neighbors.info.first;
        m_pt1 = neighbors.info.second;
        m_ray = tf::make_ray_between_points(pt0, m_pt1);

        // Initialize closest points for animation visualization
        m_closest_pt0 = pt0;
        m_closest_pt1 = m_pt1;
        m_has_closest_points = true;

        tf::point<double, 3> old_focal = {focal_point[0], focal_point[1],
                                          focal_point[2]};
        tf::point<double, 3> new_focal = m_pt1;
        m_focal_ray = tf::make_ray_between_points(old_focal, new_focal);
        m_prev_pt = m_ray.origin;
        tf::tick();
        return 0.0f + eps;
      } else if (dt < 1) {
        float t = std::min(1.f, 1.75f * tf::tock() / 1000);
        auto t_use = std::min(t, 1.f - eps);
        auto s_t = 3 * t_use * t_use - 2 * t_use * t_use * t_use;
        auto pt = m_ray.origin + s_t * m_ray.direction;
        move_instance(*selected_instance, pt - m_prev_pt);
        auto focal = m_focal_ray(s_t);
        lambda_set_focal(focal[0], focal[1], focal[2]);
        m_prev_pt = pt;

        // Update closest point visualization (shrinking line)
        m_closest_pt0 = pt;
        // m_closest_pt1 stays at target

        return t;
      }
      // Animation complete - hide visualization
      m_has_closest_points = false;
      lambda_set_focal(m_pt1[0], m_pt1[1], m_pt1[2]);
      return 1.0f;
    }
    return 2.0f;
  }

  auto set_active_color(std::size_t) -> void {
    // Don't change color on selection - keep normal appearance
  }

  auto randomize_rotations() -> void {
    for (auto &inst : bridge->get_instances()) {
      tf::vector<double, 3> at{inst.matrix[3], inst.matrix[7], inst.matrix[11]};
      auto tr = tf::random_transformation(at);
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
          inst.matrix[i * 4 + j] = tr(i, j);
        }
      }
      inst.update_frame();
    }
  }

public:
  auto get_value() -> bool { return selected_mode; }

  auto is_dragging() -> bool { return selected_mode && selected_instance.has_value(); }

  // Returns the stored closest points (computed during drag)
  auto get_closest_points() -> std::array<float, 6> {
    return {
      static_cast<float>(m_closest_pt0[0]), static_cast<float>(m_closest_pt0[1]), static_cast<float>(m_closest_pt0[2]),
      static_cast<float>(m_closest_pt1[0]), static_cast<float>(m_closest_pt1[1]), static_cast<float>(m_closest_pt1[2])
    };
  }

  auto has_valid_closest_points() -> bool { return m_has_closest_points; }

  auto get_aabb_diagonal() -> float { return m_aabb_diagonal; }

  auto set_aabb_diagonal(float diag) -> void { m_aabb_diagonal = diag; }

  // Set instance matrix from JS (for repositioning based on screen orientation)
  auto set_instance_matrix(std::size_t instance_id, const std::array<double, 16> &matrix) -> void {
    if (instance_id >= bridge->get_instances().size()) return;
    auto &inst = bridge->get_instance(instance_id);
    inst.matrix = matrix;
    inst.matrix_updated = true;
    bridge->update_frame(instance_id);
  }

  // Compute closest points for initial display (called after loading)
  auto compute_initial_closest_points() -> void {
    if (auto *pos_bridge = static_cast<positioning_bridge *>(bridge.get())) {
      if (pos_bridge->get_instances().size() < 2) return;
      // Use instance 0 as reference
      if (!pos_bridge->intersects_other(0)) {
        tf::tick();
        auto neighbors = pos_bridge->closest_metric_point_pair(0);
        add_position_time(tf::tock());
        m_closest_pt0 = neighbors.info.first;
        m_closest_pt1 = neighbors.info.second;
        m_has_closest_points = true;
      }
    }
  }

  auto OnLeftButtonUpCustom(std::array<double, 3> focal_point,
                            emscripten::val lambda_set_focal, float dt)
      -> float {
    if (get_value() || moving_mode) {
      selected_mode = false;
      moving_mode = true;
      auto new_t = position_them(focal_point, lambda_set_focal, dt);
      if (new_t == 1 && selected_instance) {
        reset_active_color(*selected_instance);
        moving_mode = false;
      }
      return new_t;
    } else if (camera_mode) {
      camera_mode = false;
    }
    moving_mode = false;
    return 2.0f;
  }

  auto OnLeftButtonDown() -> bool override {
    if (moving_mode) {
      return true;
    }
    if (selected_instance) {
      selected_mode = true;
      return true;
    } else {
      camera_mode = true;
    }
    return false;
  }

  auto OnMouseMove(std::array<float, 3> origin, std::array<float, 3> direction,
                   std::array<float, 3> camera_position,
                   std::array<float, 3> camera_focal_point) -> bool override {
    if (moving_mode) {
      return true;
    }
    tf::ray<float, 3> ray{origin, direction};
    if (!selected_mode && !camera_mode) {
      auto [instance_id, point] = bridge->ray_hit(ray);
      if (instance_id) {
        make_moving_plane(point, camera_position, camera_focal_point);
        if (selected_instance != instance_id) {
          if (selected_instance) {
            reset_active_color(*selected_instance);
          }
          set_active_color(*instance_id);
        }
        last_point = point;
      } else if (selected_instance) {
        reset_active_color(*selected_instance);
      }
      selected_instance = instance_id;
      return true;
    } else if (selected_mode && selected_instance) {
      auto next_point = tf::ray_hit(ray, moving_plane).point;
      dx = next_point - last_point;
      last_point = next_point;
      cursor_interactor_interface::move_selected(*selected_instance);

      // Update closest points for visualization (with timing)
      if (auto *pos_bridge = static_cast<positioning_bridge *>(bridge.get())) {
        if (!pos_bridge->intersects_other(*selected_instance)) {
          tf::tick();
          auto neighbors = pos_bridge->closest_metric_point_pair(*selected_instance);
          add_position_time(tf::tock());
          m_closest_pt0 = neighbors.info.first;
          m_closest_pt1 = neighbors.info.second;
          m_has_closest_points = true;
        } else {
          m_has_closest_points = false;
        }
      }
      return true;
    } else if (camera_mode) {
      return false;
    }
    return false;
  }

  auto OnKeyPress(std::string key) -> bool override {
    if (key == "n") {
      randomize_rotations();
      return true;
    } else {
      return false;
    }
  }
};

int run_main_positioning(std::vector<std::string> &paths) {
  if (paths.empty()) {
    throw std::runtime_error("At least one STL path is required.");
  }

  interactor = std::make_unique<cursor_interactor_positioning>();
  std::size_t total_polygons = 0;

  for (int i = 0; i < 2; ++i) {
    auto path_index = i < static_cast<int>(paths.size())
                          ? i
                          : static_cast<int>(paths.size() - 1);
    const auto &path = paths[path_index];
    auto poly = tf::read_stl<int>(path);
    if (!poly.size()) {
      throw std::runtime_error("Failed to read file " + path);
    }
    utils::center_and_scale_p(poly);
    total_polygons += poly.size();

    // Compute AABB diagonal for sizing
    auto aabb = tf::aabb_from(poly.points());
    float diag = tf::distance(aabb.min, aabb.max);

    auto mesh_id = interactor->add_mesh_data(std::move(poly), false);
    auto instance_id = interactor->add_instance(mesh_id);
    auto &inst = interactor->get_instances()[instance_id];
    utils::set_at(inst.matrix, {i * 15.f, (i + 2) * 15.f, 0.f});
    inst.update_frame();

    if (auto *pos_interactor =
            dynamic_cast<cursor_interactor_positioning *>(interactor.get())) {
      pos_interactor->reset_active_color(instance_id);
      // Store max AABB diagonal
      if (diag > pos_interactor->get_aabb_diagonal()) {
        pos_interactor->set_aabb_diagonal(diag);
      }
    }
  }

  // Compute initial closest points for visualization
  if (auto *pos_interactor =
          dynamic_cast<cursor_interactor_positioning *>(interactor.get())) {
    pos_interactor->compute_initial_closest_points();
  }

  interactor->total_polygons = total_polygons;
  return 0;
}

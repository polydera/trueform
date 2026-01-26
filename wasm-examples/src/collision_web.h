#pragma once
#include "trueform/core/curves_buffer.hpp"
#include "trueform/io/read_stl.hpp"
#include "trueform/random.hpp"
#include "trueform/core/form.hpp"
#include "trueform/spatial/policy/tree.hpp"
#include "trueform/spatial/ray_cast.hpp"
#include "trueform/trueform.hpp"
#include "main.h"
#include "utils/bridge_web.h"
#include "utils/cursor_interactor_interface.h"
#include "utils/utils.h"
#include <filesystem>
#include <set>
#include <string>
#include <string_view>
#include <emscripten/bind.h>
#include <emscripten/val.h>

class tf_bridge_collision : public tf_bridge_interface {
public:
  auto intersects_any(std::size_t selected_id, std::set<std::size_t> &colliding)
      -> void {
    auto &selected_inst = instances[selected_id];
    auto &selected_data = mesh_data_store[selected_inst.mesh_data_id];
    auto form0 = selected_data.polygons.polygons() |
                 tf::tag(selected_data.tree) | tf::tag(selected_inst.frame);

    for (std::size_t i = 0; i < instances.size(); ++i) {
      if (i == selected_id) {
        continue;
      }
      auto &inst = instances[i];
      auto &data = mesh_data_store[inst.mesh_data_id];
      const bool collision = tf::intersects(
          form0, data.polygons.polygons() | tf::tag(data.tree) | tf::tag(inst.frame));
      if (collision) {
        colliding.insert(i);
      } else {
        colliding.erase(i);
      }
    }
  }
};

class cursor_interactor_collision : public cursor_interactor_interface {
public:
  cursor_interactor_collision()
      : cursor_interactor_interface(
            std::make_unique<tf_bridge_collision>()) {}

private:
  std::vector<float> pick_times;
  std::vector<float> collide_times;
  std::array<double, 3> normal_mesh_color{0.8, 0.8, 0.8};
  std::array<double, 3> coliding_mesh_color{0.7, 1, 1};
  std::set<std::size_t> colliding;

  auto add_pick_time(float t) { m_pick_time = add_time(pick_times, t); }

  auto add_collide_time(float t) { m_time = add_time(collide_times, t); }

  auto handle_collisions() -> void {
    tf::tick();
    if (auto *collision_bridge =
            dynamic_cast<tf_bridge_collision *>(bridge.get())) {
      if (!selected_instance) {
        return;
      }
      collision_bridge->intersects_any(*selected_instance, colliding);
      add_collide_time(tf::tock());
      auto &instances = bridge->get_instances();
      for (std::size_t i = 0; i < instances.size(); ++i) {
        if (i == *selected_instance) {
          continue;
        }
        if (colliding.find(i) == colliding.end()) {
          reset_active_color(i);
        } else {
          set_colliding_color(i);
        }
      }
    }
  }

public:
  auto reset_active_color(std::size_t instance_id) -> void {
    bridge->get_instance(instance_id)
        .set_color(normal_mesh_color[0], normal_mesh_color[1],
                   normal_mesh_color[2]);
  }

  auto set_active_color(std::size_t) -> void {
    // Don't change color on selection - keep normal appearance
  }

  auto reset_colliding_colors() -> void {
    colliding.clear();
    auto &instances = bridge->get_instances();
    for (std::size_t i = 0; i < instances.size(); ++i) {
      if (!selected_instance || i != *selected_instance) {
        reset_active_color(i);
      }
    }
  }

  auto set_colliding_color(std::size_t instance_id) -> void {
    bridge->get_instance(instance_id)
        .set_color(coliding_mesh_color[0], coliding_mesh_color[1],
                   coliding_mesh_color[2]);
  }

public:
  auto OnLeftButtonUp() -> bool override {
    if (selected_mode) {
      selected_mode = false;
      reset_colliding_colors();
      return true;
    } else if (camera_mode) {
      camera_mode = false;
    }
    return false;
  }

  auto OnMouseMove(std::array<float, 3> origin, std::array<float, 3> direction,
                   std::array<float, 3> camera_position,
                   std::array<float, 3> camera_focal_point) -> bool override {
    tf::ray<float, 3> ray{origin, direction};
    if (!selected_mode && !camera_mode) {
      tf::tick();
      auto [instance_id, point] = bridge->ray_hit(ray);
      add_pick_time(tf::tock());
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
      move_selected(*selected_instance);
      handle_collisions();
      return true;
    } else if (camera_mode) {
      return false;
    }
    return false;
  }
};

int run_main_collisions(std::vector<std::string> &paths) {
  if (paths.empty()) {
    throw std::runtime_error("Collisions demo expects STL input paths.");
  }

  interactor = std::make_unique<cursor_interactor_collision>();

  // Load single mesh data (use first path only)
  auto poly = tf::read_stl<int>(paths[0]);
  if (!poly.size()) {
    throw std::runtime_error("Failed to load collision mesh.");
  }
  utils::center_and_scale_p(poly);
  auto mesh_id = interactor->add_mesh_data(std::move(poly), false);

  // Create 5x5 grid of instances, all sharing the same mesh_data
  int n_actors_in_dim = 5;
  std::size_t polygons_per_mesh =
      interactor->get_mesh_data_store()[mesh_id].polygons.size();

  for (int i = 0; i < n_actors_in_dim; ++i) {
    for (int j = 0; j < n_actors_in_dim; ++j) {
      auto instance_id = interactor->add_instance(mesh_id);
      auto &inst = interactor->get_instances()[instance_id];
      utils::set_at(inst.matrix, {i * 15.f, j * 15.f, 0.f});
      inst.update_frame();

      if (auto *collision_interactor =
              dynamic_cast<cursor_interactor_collision *>(interactor.get())) {
        collision_interactor->reset_active_color(instance_id);
      }
    }
  }
  interactor->total_polygons = polygons_per_mesh * n_actors_in_dim * n_actors_in_dim;
  return 0;
}

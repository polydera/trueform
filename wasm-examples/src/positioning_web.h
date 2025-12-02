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
    if (actors.size() < 2) {
      return id0;
    }
    return (id0 + 1) % actors.size();
  }

public:
  auto closest_metric_point_pair(mesh_object *actor0) {
    if (!actor0 || actors.size() < 2) {
      throw std::runtime_error("Need at least two meshes for positioning.");
    }
    std::size_t id0 = map[actor0];
    std::size_t id1 = other_index(id0);
    auto form0 = tf::make_form(frames[id0], trees[id0], polys[id0]->polygons());
    auto form1 = tf::make_form(frames[id1], trees[id1], polys[id1]->polygons());
    return tf::neighbor_search(form0, form1);
  }

  auto intersects_other(mesh_object *actor0) {
    if (!actor0 || actors.size() < 2) {
      return false;
    }
    std::size_t id0 = map[actor0];
    std::size_t id1 = other_index(id0);
    auto form0 = tf::make_form(frames[id0], trees[id0], polys[id0]->polygons());
    auto form1 = tf::make_form(frames[id1], trees[id1], polys[id1]->polygons());
    return tf::intersects(form0, form1);
  }
};

class cursor_interactor_positioning : public cursor_interactor_interface {
public:
  cursor_interactor_positioning()
      : cursor_interactor_interface(std::make_unique<positioning_bridge>()) {}

  auto reset_active_color(mesh_object *actor) -> void {
    if (!actor)
      return;
    actor->set_color(normal_mesh_color[0], normal_mesh_color[1],
                     normal_mesh_color[2]);
  }

private:
  tf::point<double, 3> normal_mesh_color{0.8, 0.8, 0.8};
  tf::point<double, 3> selected_mesh_color{1., 0.9, 1.};
  std::vector<float> positioning_times;

  bool moving_mode = false;
  tf::point<double, 3> m_pt1;
  tf::point<double, 3> m_prev_pt;
  tf::ray<float, 3> m_ray;
  tf::ray<float, 3> m_focal_ray;

  auto add_position_time(float t) {
    auto avg = add_time(positioning_times, t);
    m_time = avg;
  }

  auto move_selected(mesh_object *actor, const tf::vector<float, 3> &delta) {
    if (!actor)
      return;
    for (int i = 0; i < 3; ++i)
      actor->matrix[i * 4 + 3] += delta[i];
    bridge->update_frame(actor);
  }

  auto position_them(std::array<double, 3> focal_point,
                     emscripten::val lambda_set_focal, float dt) -> float {
    if (auto *pB = static_cast<positioning_bridge *>(bridge.get())) {
      auto eps = 0.01f;
      if (dt == 0) {
        if (pB->intersects_other(selected_actor)) {
          return 1.0f;
        }
        auto neighbors = pB->closest_metric_point_pair(selected_actor);
        auto pt0 = neighbors.info.first;
        m_pt1 = neighbors.info.second;
        m_ray = tf::make_ray_between_points(pt0, m_pt1);

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
        auto color = selected_mesh_color +
                     t_use * (normal_mesh_color - selected_mesh_color);
        move_selected(selected_actor, pt - m_prev_pt);
        auto focal = m_focal_ray(s_t);
        selected_actor->set_color(color[0], color[1], color[2]);
        lambda_set_focal(focal[0], focal[1], focal[2]);
        m_prev_pt = pt;
        return t;
      }
      lambda_set_focal(m_pt1[0], m_pt1[1], m_pt1[2]);
      return 1.0f;
    }
    return 2.0f;
  }

  auto set_active_color(mesh_object *actor) -> void {
    if (!actor)
      return;
    actor->set_color(selected_mesh_color[0], selected_mesh_color[1],
                     selected_mesh_color[2]);
  }

  auto randomize_rotations() {
      for (std::unique_ptr<mesh_object> &actor : bridge->get_actors()) {
        tf::vector<double, 3> at{actor->matrix[3], actor->matrix[7],
                                 actor->matrix[11]};
        auto tr = tf::random_transformation(at);
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 4; ++j) {
            actor->matrix[i * 4 + j] = tr(i, j);
          }
        }
        bridge->update_frame(actor.get());
      }
    }

public:
  auto get_value() -> bool { return selected_mode; }

  auto OnLeftButtonUpCustom(std::array<double, 3> focal_point,
                            emscripten::val lambda_set_focal, float dt)
      -> float {
    if (get_value() || moving_mode) {
      selected_mode = false;
      moving_mode = true;
      auto new_t = position_them(focal_point, lambda_set_focal, dt);
      if (new_t == 1) {
        reset_active_color(selected_actor);
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
    if (moving_mode)
      return true;
    if (selected_actor) {
      selected_mode = true;
      return true;
    } else {
      camera_mode = true;
    }
    return false;
  }

  auto OnMouseMove(std::array<float, 3> origin,
                   std::array<float, 3> direction,
                   std::array<float, 3> camera_position,
                   std::array<float, 3> camera_focal_point) -> bool override {
    if (moving_mode)
      return true;
    tf::ray<float, 3> ray{origin, direction};
    if (!selected_mode && !camera_mode) {
      auto [actor, point] = bridge->ray_hit(ray);
      if (actor) {
        make_moving_plane(point, camera_position, camera_focal_point);
        if (selected_actor != actor) {
          reset_active_color(selected_actor);
          set_active_color(actor);
        }
        last_point = point;
      } else {
        reset_active_color(selected_actor);
      }
      selected_actor = actor;
      return true;
    } else if (selected_mode) {
      auto next_point = tf::ray_hit(ray, moving_plane).point;
      dx = next_point - last_point;
      last_point = next_point;
      cursor_interactor_interface::move_selected(selected_actor);
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
    auto actor = std::make_unique<mesh_object>();
    actor->poly_object = std::move(poly);
    utils::set_at(actor->matrix, {i * 15.f, (i + 2) * 15.f, 0.f});
    total_polygons += actor->poly_object.size();
    interactor->push_back(std::move(actor));
    if (auto *pI =
            dynamic_cast<cursor_interactor_positioning *>(interactor.get())) {
      pI->reset_active_color(interactor->get_actors().back().get());
    }
  }

  interactor->total_polygons = total_polygons;
  return 0;
}

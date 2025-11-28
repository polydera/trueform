#pragma once
#include "trueform/core/curves_buffer.hpp"
#include "trueform/io/read_stl.hpp"
#include "trueform/random.hpp"
#include "trueform/spatial/form.hpp"
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
  void intersects_any(mesh_object *actor, std::set<mesh_object *> &colliding) {
    if (!actor) {
      return;
    }
    const auto self_index = map[actor];
    auto form0 =
        tf::make_form(frames[self_index], trees[self_index],
                      polys[self_index]->polygons());
    for (int i = 0; i < static_cast<int>(polys.size()); ++i) {
      if (i == self_index) {
        continue;
      }
      const bool collision = tf::intersects(
          form0, tf::make_form(frames[i], trees[i], polys[i]->polygons()));
      if (collision) {
        colliding.insert(actors[i].get());
      } else {
        colliding.erase(actors[i].get());
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
  std::array<double, 3> coliding_mesh_color{0.8, 1, 1};
  std::array<double, 3> selected_mesh_color{1, 0.9, 1};
  std::set<mesh_object *> colliding;

  auto add_pick_time(float t) {
    m_pick_time = add_time(pick_times, t);
  }

  auto add_collide_time(float t) {
    m_time = add_time(collide_times, t);
  }

  auto handle_collisions() {
    tf::tick();
    if (auto *pB = dynamic_cast<tf_bridge_collision *>(bridge.get())) {
      if (!selected_actor) {
        return;
      }
      pB->intersects_any(selected_actor, colliding);
      add_collide_time(tf::tock());
      for (auto &actor : bridge->get_actors()) {
        if (actor.get() == selected_actor)
          continue;
        if (colliding.find(actor.get()) == colliding.end()) {
          reset_active_color(actor.get());
        } else {
          set_colliding_color(actor.get());
        }
      }
    }
  }

public:
  auto reset_active_color(mesh_object *actor)-> void {
    if (!actor) {
      return;
    }
    actor->set_color(normal_mesh_color[0], normal_mesh_color[1],
                     normal_mesh_color[2]);
  }

  void set_active_color(mesh_object *actor) {
    if (!actor) {
      return;
    }
    actor->set_color(selected_mesh_color[0], selected_mesh_color[1],
                     selected_mesh_color[2]);
  }

  auto reset_colliding_colors() -> void {
    colliding.clear();
    for (auto &actor : bridge->get_actors()) {
      if (actor.get() != selected_actor)
        reset_active_color(actor.get());
    }
  }

  auto set_colliding_color(mesh_object *actor) -> void {
    if (!actor) {
      return;
    }
    actor->set_color(coliding_mesh_color[0], coliding_mesh_color[1],
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

  auto OnMouseMove(std::array<float, 3> origin,
                   std::array<float, 3> direction,
                   std::array<float, 3> camera_position,
                   std::array<float, 3> camera_focal_point) -> bool override {
    tf::ray<float, 3> ray{origin, direction};
    if (!selected_mode && !camera_mode) {
      tf::tick();
      auto [actor, point] = bridge->ray_hit(ray);
      add_pick_time(tf::tock());
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
      move_selected(selected_actor);
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
  std::vector<std::unique_ptr<mesh_object>> polys;
  std::vector<tf::tree<int, float, 3>> trees;

  for (int i = 0; i < static_cast<int>(paths.size()); ++i) {
    auto poly = tf::read_stl<int>(paths[i]);
    if (!poly.size())
      continue;
    utils::center_and_scale_p(poly);

    auto actor = std::make_unique<mesh_object>();
    actor->poly_object = std::move(poly);
    polys.push_back(std::move(actor));

    trees.emplace_back();
    trees.back().build(polys.back()->poly_object.polygons(),
                       tf::config_tree(4, 4));
  }

  if (polys.empty()) {
    throw std::runtime_error("Failed to load any collision meshes.");
  }

  interactor = std::make_unique<cursor_interactor_collision>();

  int n_actors_in_dim = 5;
  std::size_t poly_index = 0;
  std::size_t total_polygons = 0;
  for (int i = 0; i < n_actors_in_dim; ++i) {
    for (int j = 0; j < n_actors_in_dim; ++j) {
      auto actor = std::make_unique<mesh_object>();
      actor->poly_object = polys[poly_index]->poly_object;
      utils::set_at(actor->matrix, {i * 15.f, j * 15.f, 0.f});
      total_polygons += actor->poly_object.size();
      interactor->push_back_with_objects(std::move(actor), trees[poly_index]);
      auto a = interactor->get_actors().back().get();
      if (auto *pI =
              dynamic_cast<cursor_interactor_collision *>(interactor.get())) {
        pI->reset_active_color(a);
      }
      poly_index = (poly_index + 1) % polys.size();
    }
  }
  interactor->total_polygons = total_polygons;
  return 0;
}

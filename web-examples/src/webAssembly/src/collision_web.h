#pragma once
#include "trueform/core/curves_buffer.hpp"
#include "trueform/io/read_stl.hpp"
#include "trueform/random.hpp"
#include "trueform/trueform.hpp"
#include "trueform/spatial/form.hpp"
#include "trueform/spatial/ray_cast.hpp"

#include "main.h"
#include "utils/utils.h"
#include "utils/bridge_web.h"
#include "utils/cursor_interactor_interface.h"

#include <filesystem>
#include <set>
#include <string>
#include <string_view>

#include <emscripten/bind.h>
#include <emscripten/val.h>

class tf_bridge_collision : public tf_bridge_interface {
public:
  auto intersects_any(mesh_object *actor, std::set<mesh_object *> &colliding) {
    std::size_t id = map[actor];
    auto form0 =
        tf::make_form(frames[id], trees[id], polys[id]->polygons());
    for (std::size_t i = 0; i < polys.size(); ++i) {
      if (i == id)
        continue;
      auto collision = tf::intersects(
          form0, tf::make_form(frames[i], trees[i], polys[i]->polygons()));
      if (collision)
        colliding.insert(actors[i].get());
      else {
        colliding.erase(actors[i].get());
      }
    }
  }
};

class cursor_interactor_collision : public cursor_interactor_interface {
public:
  cursor_interactor_collision() : cursor_interactor_interface(std::make_unique<tf_bridge_collision>()) {}

public:
  std::unique_ptr<mesh_object> result_mesh = std::make_unique<mesh_object>();
  std::unique_ptr<mesh_object> curve_mesh = std::make_unique<mesh_object>();

private:
  std::vector<float> pick_times;
  std::vector<float> collide_times;
  std::array<double, 3> normal_mesh_color{0.8, 0.8, 0.8};
  std::array<double, 3> coliding_mesh_color{0.8, 1, 1};
  std::array<double, 3> selected_mesh_color{1, 0.9, 1};
  std::set<mesh_object *> colliding;


  auto add_pick_time(float t) {
    auto pick_time = add_time(pick_times, t);
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "Picking time per frame: %.1f mcs",
                  pick_time * 1000);
    std::cout << buffer << std::endl;
    m_pick_time = pick_time;
  }

  auto add_collide_time(float t) {
    auto collide_time = add_time(collide_times, t);
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "Collision time per frame: %.1f mcs",
                  collide_time * 1000);
    std::cout << buffer << std::endl;
    m_time = collide_time;
  }

  auto handle_collisions() {
    tf::tick();
    if(auto pB = static_cast<tf_bridge_collision*>(bridge.get())) {
      pB->intersects_any(selected_actor, colliding);
      add_collide_time(tf::tock());
      for (auto &actor : pB->get_actors()) {
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
  auto reset_active_color(mesh_object *selected_actor) -> void {
    if (!selected_actor)
      return;
    selected_actor->set_color(
        normal_mesh_color[0], normal_mesh_color[1], normal_mesh_color[2]);
  }

  auto set_active_color(mesh_object *selected_actor) -> void {
    selected_actor->set_color(
        selected_mesh_color[0], selected_mesh_color[1], selected_mesh_color[2]);
  }

  auto reset_colliding_colors() -> void {
    colliding.clear();
    if(auto pB = static_cast<tf_bridge_collision*>(bridge.get())) {
      for (auto &actor : pB->get_actors()) {
        if (actor.get() != selected_actor)
          reset_active_color(actor.get());
      }
    }
  }

  auto set_colliding_color(mesh_object *selected_actor) -> void {
    selected_actor->set_color(
        coliding_mesh_color[0], coliding_mesh_color[1], coliding_mesh_color[2]);
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

  auto OnMouseMove(std::array<float, 3> origin, std::array<float, 3> direction, std::array<float, 3> cameraPosition, std::array<float, 3> cameraFocalPoint) -> bool override {
    tf::ray<float, 3> ray{origin, direction};
    if (!selected_mode && !camera_mode) {
      tf::tick();
      auto [actor, point] = bridge->ray_hit(ray);
      add_pick_time(tf::tock());
      if (actor) {
        make_moving_plane(point, cameraPosition, cameraFocalPoint);
        if (selected_actor != actor) {
          reset_active_color(selected_actor);
          set_active_color(actor);
        }
        this->last_point = point;
      } else {
        reset_active_color(selected_actor);
      }
      selected_actor = actor;
      return true;
    } else if (selected_mode) {
      auto next_point = tf::ray_hit(ray, moving_plane).point;
      dx = next_point - this->last_point;
      this->last_point = next_point;
      move_selected(selected_actor);
      handle_collisions();
      return true;
    } else if (camera_mode) {
        return false;
    }
    return false;
  }
};

int run_main_collisions(std::vector<std::string>& paths) {
  std::vector<std::unique_ptr<mesh_object>> polys;

  for (int i = 0; i < static_cast<int>(paths.size()); ++i) {
    auto poly = tf::read_stl<int>(paths[i]);
    std::cout << "run main 0: " << poly.size() << std::endl;
    if (!poly.size())
      continue;
    utils::center_and_scale_p(poly);

    auto actor = std::make_unique<mesh_object>();
    actor->poly_object = std::move(poly);
    polys.push_back(std::move(actor));
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
      interactor->push_back(std::move(actor));
      auto a = interactor->get_actors().back().get();
      if(auto pI = dynamic_cast<cursor_interactor_collision*>(interactor.get()))
        pI->reset_active_color(a);
      poly_index = (poly_index + 1) % polys.size();
    }
  }
  interactor->total_polygons = total_polygons;
  return 0;
}
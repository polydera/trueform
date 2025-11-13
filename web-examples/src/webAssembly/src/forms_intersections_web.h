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
#include <string>
#include <string_view>

#include <emscripten/bind.h>
#include <emscripten/val.h>

class tf_bridge_forms_intersections : public tf_bridge_interface {
public:
  auto compute_intersection_curves()  {
    auto form0 =
        tf::make_form(frames[0], trees[0], polys[0]->polygons()) //
        | tf::tag(face_memberships[0])                          //
        | tf::tag(manifold_edge_links[0]);

    auto form1 =
        tf::make_form(frames[1], trees[1], polys[1]->polygons()) //
        | tf::tag(face_memberships[1])                          //
        | tf::tag(manifold_edge_links[1]);
    return tf::make_intersection_curves(form0, form1);
  }
};

class cursor_interactor_forms_intersections : public cursor_interactor_interface {
public:
  cursor_interactor_forms_intersections() : cursor_interactor_interface(std::make_unique<tf_bridge_forms_intersections>()) {}

public:
  std::unique_ptr<mesh_object> curve_mesh = std::make_unique<mesh_object>();

private:
  std::vector<float> intersection_times;

  auto add_intersection_time(float t) {
    auto intersection_time = add_time(intersection_times, t);
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer),
                  "Intersection curve time per scroll: %.1f ms", intersection_time);
    std::cout << buffer << std::endl;
    m_time = intersection_time;
  }

  auto compute_curves() {
    tf::tick();
    if(auto pB = static_cast<tf_bridge_forms_intersections*>(bridge.get())) {
      auto curves = pB->compute_intersection_curves();
      add_intersection_time(tf::tock());
      curve_mesh->set_curves_object(std::move(curves));
    }
  }

  auto randomize_rotations() {
    for (std::unique_ptr<mesh_object>& actor : bridge->get_actors()) {
      tf::vector<double, 3> at{actor->matrix[3], actor->matrix[7],
                               actor->matrix[11]};
      auto tr = tf::random_transformation(at);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
          actor->matrix[i*4 + j] = tr(i, j);
      bridge->update_frame(actor.get());
    }
  }

public:
  auto OnMouseMove(std::array<float, 3> origin, std::array<float, 3> direction, std::array<float, 3> cameraPosition, std::array<float, 3> cameraFocalPoint) -> bool override {
    tf::ray<float, 3> ray{origin, direction};
    if (!selected_mode && !camera_mode) {
      auto [actor, point] = bridge->ray_hit(ray);
      if (actor) {
        make_moving_plane(point, cameraPosition, cameraFocalPoint);
        this->last_point = point;
      }
      selected_actor = actor;
      return true;
    } else if (selected_mode) {
      auto next_point = tf::ray_hit(ray, moving_plane).point;
      dx = next_point - this->last_point;
      this->last_point = next_point;
      move_selected(selected_actor);
      compute_curves();
      return true;
    } else if (camera_mode) {
        return false;
    }
    return false;
  }

  auto OnKeyPress(std::string key) -> bool override  {
    if (key == "n") {
      randomize_rotations();
      compute_curves();
      return true;
    } else {
      return false;
    }
  }
};

int run_main_forms_intersections(std::vector<std::string>& paths) {
  if (paths.size() < 2) {
    std::cerr << "forms_intersections requires at least 2 files\n";
    return 1;
  }

  std::vector<std::unique_ptr<mesh_object>> polys;

  for (int i = 0; i < 2 && i < static_cast<int>(paths.size()); ++i) {
    auto poly = tf::read_stl<int>(paths[i]);
    std::cout << "Reading file " << i << ": " << poly.size() << " polygons" << std::endl;
    if (!poly.size())
      continue;
    utils::center_and_scale_p(poly);

    auto actor = std::make_unique<mesh_object>();
    actor->poly_object = std::move(poly);
    polys.push_back(std::move(actor));
  }

  if (polys.size() < 2) {
    std::cerr << "Need at least 2 valid meshes for intersection curves\n";
    return 1;
  }

  interactor = std::make_unique<cursor_interactor_forms_intersections>();

  std::size_t total_polygons = 0;
  for (int i = 0; i < 2; ++i) {
    auto actor = std::make_unique<mesh_object>();
    actor->poly_object = polys[i]->poly_object;
    utils::set_at(actor->matrix, {i * 15.f, 0.f, 0.f});
    total_polygons += actor->poly_object.size();
    interactor->push_back(std::move(actor));
  }
  interactor->total_polygons = total_polygons;
  return 0;
}
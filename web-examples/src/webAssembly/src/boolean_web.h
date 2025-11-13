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

class tf_bridge : public tf_bridge_interface {
public:
  auto compute_boolean() {
    auto form0 = tf::make_form(frames[0], trees[0], polys[0]->polygons()) //
                 | tf::tag(face_memberships[0])                         //
                 | tf::tag(manifold_edge_links[0]);

    auto form1 = tf::make_form(frames[1], trees[1], polys[1]->polygons()) //
                 | tf::tag(face_memberships[1])                         //
                 | tf::tag(manifold_edge_links[1]);
    return tf::make_boolean(form0, form1, tf::boolean_op::left_difference,
                            tf::return_curves);
  }
};

class cursor_interactor : public cursor_interactor_interface {
public:
  cursor_interactor() : cursor_interactor_interface(std::make_unique<tf_bridge>()) {}

public:
  std::unique_ptr<mesh_object> result_mesh = std::make_unique<mesh_object>();
  std::unique_ptr<mesh_object> curve_mesh = std::make_unique<mesh_object>();

private:
  std::vector<float> boolean_times;

  auto add_boolean_time(float t) {
    auto boolean_time = add_time(boolean_times, t);
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "Boolean time per frame  : %.1f ms",
                  boolean_time);
    m_time = boolean_time;
  }

  auto compute_curves() {
    tf::tick();
    if(auto pB = static_cast<tf_bridge*>(bridge.get())) {
      auto [res_mesh, labels, curves] = pB->compute_boolean();
      add_boolean_time(tf::tock());
      (void)labels;
      result_mesh->set_polydata(std::move(res_mesh));
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

int run_main(std::vector<std::string>& paths) {
  auto path = paths[0];
  std::cout << "Reading file: " << path << std::endl;
  auto poly = tf::read_stl<int>(path);
  std::cout << "run main 0: " << poly.size() << std::endl;
  if(paths.size() == 2) {
	path = paths[1];
  }
  auto poly2 = tf::read_stl<int>(path);
  if (!poly.size()) {
    std::cout << "Failed to read file" << std::endl;
    throw std::runtime_error("Failed to read file");
  }
  interactor = std::make_unique<cursor_interactor>();

  utils::center_and_scale_p(poly);
  auto actor = std::make_unique<mesh_object>();
  actor->poly_object = std::move(poly);
  utils::set_at(actor->matrix, {0 * 15.f, 0.f, 0.f});
  interactor->push_back(std::move(actor));

  utils::center_and_scale_p(poly2);
  auto actor2 = std::make_unique<mesh_object>();
  actor2->poly_object = std::move(poly2);
  utils::set_at(actor2->matrix, {1 * 15.f, 0.f, 0.f});
  interactor->push_back(std::move(actor2));
  return 0;
}
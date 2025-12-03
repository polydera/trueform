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
#include <string>
#include <string_view>
#include <emscripten/bind.h>
#include <emscripten/val.h>

class tf_bridge_forms_intersections : public tf_bridge_interface {
public:
  auto compute_intersection_curves() {
    auto &inst0 = instances[0];
    auto &inst1 = instances[1];
    auto &data0 = mesh_data_store[inst0.mesh_data_id];
    auto &data1 = mesh_data_store[inst1.mesh_data_id];

    auto form0 = tf::make_form(inst0.frame, data0.tree, data0.polygons.polygons()) |
                 tf::tag(*data0.face_membership) |
                 tf::tag(*data0.manifold_edge_link);

    auto form1 = tf::make_form(inst1.frame, data1.tree, data1.polygons.polygons()) |
                 tf::tag(*data1.face_membership) |
                 tf::tag(*data1.manifold_edge_link);

    return tf::make_intersection_curves(form0, form1);
  }
};

class cursor_interactor_forms_intersections
    : public cursor_interactor_interface {
public:
  cursor_interactor_forms_intersections()
      : cursor_interactor_interface(
            std::make_unique<tf_bridge_forms_intersections>()) {}

private:
  std::vector<float> intersection_times;

  auto add_intersection_time(float t) -> void {
    m_time = add_time(intersection_times, t);
  }

  auto compute_curves() -> void {
    tf::tick();
    if (auto *intersect_bridge =
            dynamic_cast<tf_bridge_forms_intersections *>(bridge.get())) {
      auto curves_result = intersect_bridge->compute_intersection_curves();
      add_intersection_time(tf::tock());
      curves.set_curves(std::move(curves_result));
    }
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
  auto OnMouseMove(std::array<float, 3> origin, std::array<float, 3> direction,
                   std::array<float, 3> camera_position,
                   std::array<float, 3> camera_focal_point) -> bool override {
    tf::ray<float, 3> ray{origin, direction};
    if (!selected_mode && !camera_mode) {
      auto [instance_id, point] = bridge->ray_hit(ray);
      if (instance_id) {
        make_moving_plane(point, camera_position, camera_focal_point);
        last_point = point;
      }
      selected_instance = instance_id;
      return true;
    } else if (selected_mode && selected_instance) {
      auto next_point = tf::ray_hit(ray, moving_plane).point;
      dx = next_point - last_point;
      last_point = next_point;
      move_selected(*selected_instance);
      compute_curves();
      return true;
    } else if (camera_mode) {
      return false;
    }
    return false;
  }

  auto OnKeyPress(std::string key) -> bool override {
    if (key == "n") {
      randomize_rotations();
      compute_curves();
      return true;
    } else {
      return false;
    }
  }
};

int run_main_forms_intersections(std::vector<std::string> &paths) {
  if (paths.size() < 2) {
    throw std::runtime_error(
        "forms_intersections requires at least two STL inputs.");
  }

  interactor = std::make_unique<cursor_interactor_forms_intersections>();

  // Load mesh data
  std::vector<std::size_t> mesh_data_ids;
  for (int i = 0; i < 2 && i < static_cast<int>(paths.size()); ++i) {
    auto poly = tf::read_stl<int>(paths[i]);
    if (!poly.size()) {
      continue;
    }
    utils::center_and_scale_p(poly);
    auto mesh_id = interactor->add_mesh_data(std::move(poly), true);
    mesh_data_ids.push_back(mesh_id);
  }

  if (mesh_data_ids.size() < 2) {
    throw std::runtime_error(
        "Need at least two valid meshes for intersection curves.");
  }

  // Create instances
  std::size_t total_polygons = 0;
  for (int i = 0; i < 2; ++i) {
    auto mesh_id = mesh_data_ids[i];
    auto instance_id = interactor->add_instance(mesh_id);
    auto &inst = interactor->get_instances()[instance_id];
    utils::set_at(inst.matrix, {i * 15.f, 0.f, 0.f});
    inst.update_frame();
    total_polygons += interactor->get_mesh_data_store()[mesh_id].polygons.size();
  }
  interactor->total_polygons = total_polygons;
  return 0;
}

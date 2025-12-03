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

class tf_bridge_boolean : public tf_bridge_interface {
public:
  auto compute_boolean() {
    auto &inst0 = instances[0];
    auto &inst1 = instances[1];
    auto &data0 = mesh_data_store[inst0.mesh_data_id];
    auto &data1 = mesh_data_store[inst1.mesh_data_id];

    const auto form0 =
        tf::make_form(inst0.frame, data0.tree, data0.polygons.polygons()) |
        tf::tag(*data0.face_membership) | tf::tag(*data0.manifold_edge_link);
    const auto form1 =
        tf::make_form(inst1.frame, data1.tree, data1.polygons.polygons()) |
        tf::tag(*data1.face_membership) | tf::tag(*data1.manifold_edge_link);

    return tf::make_boolean(form0, form1, tf::boolean_op::left_difference,
                            tf::return_curves);
  }
};

class cursor_interactor_boolean final : public cursor_interactor_interface {
public:
  cursor_interactor_boolean()
      : cursor_interactor_interface(std::make_unique<tf_bridge_boolean>()) {}

private:
  std::vector<float> boolean_times;

  auto add_boolean_time(float t) -> void {
    auto boolean_time = add_time(boolean_times, t);
    m_time = boolean_time;
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
  auto compute_curves() -> void {
    tf::tick();
    if (auto *boolean_bridge =
            dynamic_cast<tf_bridge_boolean *>(bridge.get())) {
      auto [res_mesh, labels, curves_result] = boolean_bridge->compute_boolean();
      add_boolean_time(tf::tock());
      (void)labels;
      result.set_polygons(std::move(res_mesh));
      curves.set_curves(std::move(curves_result));
    }
  }

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

inline auto load_centered_mesh_data(cursor_interactor_interface *interactor,
                                    const std::string &path) -> std::size_t {
  auto poly = tf::read_stl<int>(path);
  if (!poly.size()) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  utils::center_and_scale_p(poly);
  return interactor->add_mesh_data(std::move(poly), true);
}

int run_main(std::vector<std::string> &paths) {
  if (paths.empty()) {
    throw std::runtime_error(
        "Boolean example expects at least one STL path argument.");
  }

  interactor = std::make_unique<cursor_interactor_boolean>();

  // Load mesh data
  auto mesh_id0 = load_centered_mesh_data(interactor.get(), paths[0]);
  auto mesh_id1 = load_centered_mesh_data(
      interactor.get(), paths.size() >= 2 ? paths[1] : paths[0]);

  // Create instances
  auto inst_id0 = interactor->add_instance(mesh_id0);
  auto inst_id1 = interactor->add_instance(mesh_id1);

  auto &inst0 = interactor->get_instances()[inst_id0];
  auto &inst1 = interactor->get_instances()[inst_id1];

  utils::set_at(inst0.matrix, {0.f, 0.f, 0.f});
  utils::set_at(inst1.matrix, {15.f, 0.f, 0.f});
  inst0.update_frame();
  inst1.update_frame();

  if (auto *boolean_interactor =
          dynamic_cast<cursor_interactor_boolean *>(interactor.get())) {
    boolean_interactor->compute_curves();
  }
  return 0;
}

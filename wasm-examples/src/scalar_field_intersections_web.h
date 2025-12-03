#pragma once
#include <iostream>
#include <stdexcept>
#include <vector>
#include "trueform/core/curves_buffer.hpp"
#include "trueform/io/read_stl.hpp"
#include "trueform/random.hpp"
#include "trueform/trueform.hpp"
#include "main.h"
#include "utils/bridge_web.h"
#include "utils/cursor_interactor_interface.h"
#include "utils/utils.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>

class scalar_field_intersections_bridge : public tf_bridge_interface {
public:
  auto compute_isocontours(const tf::buffer<float> &scalars,
                           const std::vector<float> &cutvalues) {
    if (mesh_data_store.empty()) {
      throw std::runtime_error(
          "No meshes available for scalar field intersections");
    }
    auto &data = mesh_data_store[0];
    return tf::make_isocontours(data.polygons.polygons(),
                                tf::make_range(scalars),
                                tf::make_range(cutvalues));
  }
};

class cursor_interactor_scalar_field_intersections
    : public cursor_interactor_interface {
public:
  cursor_interactor_scalar_field_intersections()
      : cursor_interactor_interface(
            std::make_unique<scalar_field_intersections_bridge>()) {}

private:
  tf::buffer<float> scalars;
  std::vector<float> intersection_times;
  float min_d = 0.0f;
  float max_d = 1.0f;
  float distance = 0.0f;

  auto add_intersection_time(float t) -> void {
    m_time = add_time(intersection_times, t);
  }

public:
  auto compute_curves() -> void {
    std::vector<float> cutvalues;
    cutvalues.reserve(20);
    for (int i = -10; i < 10; ++i) {
      cutvalues.push_back(distance + static_cast<float>(i) * 0.5f);
    }

    tf::tick();
    if (auto *scalar_bridge =
            dynamic_cast<scalar_field_intersections_bridge *>(bridge.get())) {
      auto curves_result = scalar_bridge->compute_isocontours(scalars, cutvalues);
      add_intersection_time(tf::tock());
      curves.set_curves(std::move(curves_result));
    }
  }

  auto reset_plane() -> void {
    auto &mesh_store = bridge->get_mesh_data_store();
    if (mesh_store.empty()) {
      return;
    }
    auto points = mesh_store[0].polygons.points();
    if (!points.size()) {
      return;
    }
    auto plane = tf::make_plane(tf::normalized(tf::random_vector<float, 3>()),
                                points[tf::random<int>(0, points.size() - 1)]);
    scalars.allocate(points.size());
    tf::parallel_transform(points, scalars, tf::distance_f(plane));
    distance = 0.0f;
    min_d = *std::min_element(scalars.begin(), scalars.end());
    max_d = *std::max_element(scalars.begin(), scalars.end());
  }

public:
  auto OnMouseMove(std::array<float, 3>, std::array<float, 3>,
                   std::array<float, 3>, std::array<float, 3>) -> bool override {
    return false;
  }

  auto OnMouseWheel(int delta, bool shiftKey) -> bool override {
    if (shiftKey) {
      distance += static_cast<float>(delta) * 0.05f;
      // Wrap for infinite scrolling
      const float range = max_d - min_d;
      float offset = std::fmod(distance - min_d, range);
      if (offset < 0) {
        offset += range;
      }
      distance = min_d + offset;
      compute_curves();
      return true;
    }
    return false;
  }

  auto OnKeyPress(std::string key) -> bool override {
    if (key == "n") {
      reset_plane();
      compute_curves();
      return true;
    }
    return false;
  }
};

int run_main_scalar_field_intersections(std::string path) {
  auto poly = tf::read_stl<int>(path);
  if (!poly.size()) {
    throw std::runtime_error("Failed to read mesh for scalar field intersections");
  }

  interactor = std::make_unique<cursor_interactor_scalar_field_intersections>();

  utils::center_and_scale_p(poly);
  auto mesh_id = interactor->add_mesh_data(std::move(poly), false);
  interactor->add_instance(mesh_id);

  if (auto *scalar_interactor =
          dynamic_cast<cursor_interactor_scalar_field_intersections *>(
              interactor.get())) {
    scalar_interactor->reset_plane();
    scalar_interactor->compute_curves();
  }
  return 0;
}

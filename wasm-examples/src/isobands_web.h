#pragma once
#include <iostream>
#include <ostream>
#include <vector>
#include "trueform/core/curves_buffer.hpp"
#include "trueform/io/read_stl.hpp"
#include "trueform/trueform.hpp"
#include "main.h"
#include "utils/bridge_web.h"
#include "utils/cursor_interactor_interface.h"
#include "utils/utils.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>

class isobands_bridge : public tf_bridge_interface {
public:
  auto compute_isobands(const tf::buffer<float> &scalars,
                        const std::vector<float> &cutvalues,
                        const std::vector<int> &selected_values) {
    auto &data = mesh_data_store[0];
    return tf::make_isobands<int>(data.polygons.polygons(), scalars,
                                  tf::make_range(cutvalues),
                                  tf::make_range(selected_values),
                                  tf::return_curves);
  }
};

class cursor_interactor_isobands : public cursor_interactor_interface {
public:
  cursor_interactor_isobands()
      : cursor_interactor_interface(std::make_unique<isobands_bridge>()) {}

private:
  tf::buffer<float> scalars;
  std::vector<float> isobands_times;
  float min_d = 0.0f;
  float max_d = 1.0f;
  float distance = 0.0f;

  auto add_isobands_time(float t) -> void { m_time = add_time(isobands_times, t); }

public:
  auto compute_curves() -> void {
    int n = 10;
    const float s = (max_d - min_d) / static_cast<float>(n);
    const float a = (distance - min_d) / s;
    int k = static_cast<int>(std::floor(a));
    if (k < 0) {
      k = 0;
    }
    if (k >= n) {
      k = n - 1;
    }
    std::vector<float> cutvalues;
    for (int i = 0; i < n; ++i) {
      cutvalues.push_back(distance + (i - k) * s);
    }
    std::vector<int> selected_values;
    const int parity = k & 1;
    for (int i = 0; i < n; ++i) {
      if ((i & 1) == parity) {
        selected_values.push_back(i);
      }
    }
    tf::tick();
    if (auto *isobands_brige = dynamic_cast<isobands_bridge *>(bridge.get())) {
      auto [polys, _, curves_result] =
          isobands_brige->compute_isobands(scalars, cutvalues, selected_values);
      add_isobands_time(tf::tock());
      result.set_polygons(std::move(polys));
      curves.set_curves(std::move(curves_result));
    }
  }

  auto reset_plane() -> tf::buffer<float> {
    auto &mesh_store = bridge->get_mesh_data_store();
    if (mesh_store.empty()) {
      throw std::runtime_error("Isobands bridge requires at least one mesh.");
    }
    auto points = mesh_store[0].polygons.points();
    auto center = tf::centroid(points);
    auto normal = tf::make_unit_vector(1.f, 2.f, 1.f);
    auto plane = tf::make_plane(normal, center);
    scalars.allocate(points.size());
    tf::parallel_transform(points, scalars, tf::distance_f(plane));
    distance = 0;
    min_d = *std::min_element(scalars.begin(), scalars.end());
    max_d = *std::max_element(scalars.begin(), scalars.end());
    return scalars;
  }

public:
  auto OnMouseMove(std::array<float, 3>, std::array<float, 3>,
                   std::array<float, 3>, std::array<float, 3>) -> bool override {
    return false;
  }

  auto OnMouseWheel(int delta, bool shiftKey) -> bool override {
    if (shiftKey) {
      const float range = max_d - min_d;
      distance += delta * 0.003f * range;
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
    } else {
      return false;
    }
  }
};

int run_main_isobands(std::string path) {
  auto poly = tf::read_stl<int>(path);
  if (!poly.size()) {
    throw std::runtime_error("Failed to read file");
  }

  interactor = std::make_unique<cursor_interactor_isobands>();

  utils::center_and_scale_p(poly);
  auto mesh_id = interactor->add_mesh_data(std::move(poly), false);
  interactor->add_instance(mesh_id);

  if (auto *isobands_interactor =
          dynamic_cast<cursor_interactor_isobands *>(interactor.get())) {
    isobands_interactor->reset_plane();
    isobands_interactor->compute_curves();
  }

  return 0;
}

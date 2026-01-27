#pragma once
#include <iostream>
#include <vector>
#include "trueform/geometry/triangulated.hpp"
#include "trueform/io/read_stl.hpp"
#include "trueform/trueform.hpp"
#include "main.h"
#include "utils/bridge_web.h"
#include "utils/cursor_interactor_interface.h"
#include "utils/utils.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>

class cross_section_bridge : public tf_bridge_interface {
public:
  auto compute_cross_section(const tf::buffer<float> &scalars, float cut_value) {
    auto &data = mesh_data_store[0];
    // Extract isocontours at the cut value
    auto curves = tf::make_isocontours(data.polygons.polygons(), tf::make_range(scalars),
                                            cut_value);
    // Triangulate the curves into filled polygons
    auto triangles = tf::triangulated<int>(
        tf::make_polygons(curves.paths(), curves.points()));
    return std::make_pair(std::move(triangles), std::move(curves));
  }
};

class cursor_interactor_cross_section : public cursor_interactor_interface {
public:
  cursor_interactor_cross_section()
      : cursor_interactor_interface(std::make_unique<cross_section_bridge>()) {}

private:
  tf::buffer<float> scalars;
  std::vector<float> cross_section_times;
  float min_d = 0.0f;
  float max_d = 1.0f;
  float cut_value = 0.0f;

  auto add_cross_section_time(float t) -> void {
    m_time = add_time(cross_section_times, t);
  }

public:
  auto compute_cross_section() -> void {
    tf::tick();
    if (auto *cs_bridge =
            dynamic_cast<cross_section_bridge *>(bridge.get())) {
      auto [triangles, curves_result] =
          cs_bridge->compute_cross_section(scalars, cut_value);
      add_cross_section_time(tf::tock());
      result.set_polygons(std::move(triangles));
      curves.set_curves(std::move(curves_result));
    }
  }

  auto reset_plane() -> tf::buffer<float> {
    auto &mesh_store = bridge->get_mesh_data_store();
    if (mesh_store.empty()) {
      throw std::runtime_error("Cross section bridge requires at least one mesh.");
    }
    auto points = mesh_store[0].polygons.points();
    auto center = tf::centroid(points);
    // Use diagonal plane with normal (1, 2, 1) - same as VTK example
    auto normal = tf::make_unit_vector(1.f, 2.f, 1.f);
    auto plane = tf::make_plane(normal, center);

    scalars.allocate(points.size());
    tf::parallel_transform(points, scalars, tf::distance_f(plane));

    min_d = *std::min_element(scalars.begin(), scalars.end());
    max_d = *std::max_element(scalars.begin(), scalars.end());
    cut_value = (min_d + max_d) * 0.5f;

    return scalars;
  }

  auto randomize_plane() -> void {
    auto &mesh_store = bridge->get_mesh_data_store();
    if (mesh_store.empty()) {
      return;
    }
    auto points = mesh_store[0].polygons.points();
    auto plane =
        tf::make_plane(tf::normalized(tf::random_vector<float, 3>()),
                       points[tf::random<int>(0, points.size() - 1)]);
    tf::parallel_transform(points, scalars, tf::distance_f(plane));
    min_d = *std::min_element(scalars.begin(), scalars.end());
    max_d = *std::max_element(scalars.begin(), scalars.end());
    cut_value = (min_d + max_d) * 0.5f;
  }

public:
  auto OnMouseMove(std::array<float, 3>, std::array<float, 3>,
                   std::array<float, 3>, std::array<float, 3>) -> bool override {
    return false;
  }

  auto OnMouseWheel(int delta, bool shiftKey) -> bool override {
    if (shiftKey) {
      const float range = max_d - min_d;
      const float margin = range * 0.01f;

      cut_value += delta * 0.003f * range;
      // Clamp to valid range
      cut_value = std::max(min_d + margin, std::min(max_d - margin, cut_value));

      compute_cross_section();
      return true;
    }
    return false;
  }

  auto OnKeyPress(std::string key) -> bool override {
    if (key == "n") {
      randomize_plane();
      compute_cross_section();
      return true;
    }
    return false;
  }
};

int run_main_cross_section(std::string path) {
  auto poly = tf::read_stl<int>(path);
  if (!poly.size()) {
    throw std::runtime_error("Failed to read file");
  }

  interactor = std::make_unique<cursor_interactor_cross_section>();

  utils::center_and_scale_p(poly);
  auto mesh_id = interactor->add_mesh_data(std::move(poly), false);
  interactor->add_instance(mesh_id);

  if (auto *cs_interactor =
          dynamic_cast<cursor_interactor_cross_section *>(interactor.get())) {
    cs_interactor->reset_plane();
    cs_interactor->compute_cross_section();
  }

  return 0;
}

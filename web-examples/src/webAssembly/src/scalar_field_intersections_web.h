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
    if (polys.empty()) {
      throw std::runtime_error("No meshes available for scalar field intersections");
    }
    return tf::make_isocontours(polys[0]->polygons(), tf::make_range(scalars),
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
  float distance = 0.0f;

  auto add_intersection_time(float t) -> void { m_time = add_time(intersection_times, t); }

public:
  auto compute_curves() -> void {
    std::vector<float> cutvalues;
    cutvalues.reserve(20);
    for (int i = -10; i < 10; ++i) {
      cutvalues.push_back(distance + static_cast<float>(i) * 0.5f);
    }

    tf::tick();
    if (auto *pB =
            static_cast<scalar_field_intersections_bridge *>(bridge.get())) {
      auto curves = pB->compute_isocontours(scalars, cutvalues);
      add_intersection_time(tf::tock());
      curve_mesh->set_curves_object(std::move(curves));
    }
  }

  auto reset_plane() -> void {
    auto &actors = bridge->get_actors();
    if (actors.empty()) {
      return;
    }
    auto points = actors[0]->poly_object.points();
    if (!points.size()) {
      return;
    }
    auto plane = tf::make_plane(tf::normalized(tf::random_vector<float, 3>()),
                                points[tf::random<int>(0, points.size() - 1)]);
    scalars.allocate(points.size());
    tf::parallel_transform(points, scalars, tf::distance_f(plane));
    distance = 0.0f;
  }

public:
  auto OnMouseMove(std::array<float, 3>, std::array<float, 3>,
                   std::array<float, 3>, std::array<float, 3>) -> bool override {
    return false;
  }

  auto OnMouseWheel(int delta, bool shiftKey) -> bool override {
    if (shiftKey) {
      distance += static_cast<float>(delta) * 0.05f;
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
  auto actor = std::make_unique<mesh_object>();
  actor->poly_object = std::move(poly);
  interactor->push_back(std::move(actor));

  if (auto *pI =
          dynamic_cast<cursor_interactor_scalar_field_intersections *>(interactor.get())) {
    pI->reset_plane();
    pI->compute_curves();
  }
  return 0;
}

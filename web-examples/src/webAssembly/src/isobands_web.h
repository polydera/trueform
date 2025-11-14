#pragma once
#include <iostream>
#include <ostream>

#include "trueform/core/curves_buffer.hpp"
#include "trueform/io/read_stl.hpp"
#include "trueform/trueform.hpp"
#include "main.h"
#include "utils/utils.h"
#include "utils/bridge_web.h"
#include "utils/cursor_interactor_interface.h"
#include <vector>
#include <emscripten/bind.h>
#include <emscripten/val.h>

class isobands_bridge : public tf_bridge_interface {
public:
  auto compute_isobands(const tf::buffer<float>& scalars,
                        const std::vector<float>& cutvalues,
                        const std::vector<int>& selected_values) {
    std::cout << "compute_isobands polys" << polys[0] << ", size: " << polys[0]->size() << std::endl;
    return tf::make_isobands<int>(polys[0]->polygons(), scalars, tf::make_range(cutvalues),
                                  tf::make_range(selected_values), tf::return_curves);
  }
};

class cursor_interactor_isobands : public cursor_interactor_interface {
public:
  cursor_interactor_isobands() : cursor_interactor_interface(std::make_unique<isobands_bridge>()) {}

private:
  tf::buffer<float> scalars;
  std::vector<float> isobands_times;
  float min_d = 0.0f;
  float max_d = 1.0f;
  float distance = 0.0f;

  auto add_isobands_time(float t) {
    auto isobands_time = add_time(isobands_times, t);
    m_time = isobands_time;
  }
public:
  auto compute_curves() {
    int N = 10;
    const float s = (max_d - min_d) / static_cast<float>(N);
    const float a = (distance - min_d) / s;
    int k = static_cast<int>(std::floor(a));
    if (k < 0) k = 0;
    if (k >= N) k = N - 1;
    std::vector<float> cutvalues;
    for (int i = 0; i < N; ++i)
      cutvalues.push_back(distance + (i - k) * s);
    std::vector<int> selected_values;
    const int parity = k & 1;
    for (int i = 0; i < N; ++i)
      if ((i & 1) == parity)
        selected_values.push_back(i);
    tf::tick();
    if(auto pB = static_cast<isobands_bridge*>(bridge.get())) {
      auto [polys, _, curves] = pB->compute_isobands(scalars, cutvalues, selected_values);
      add_isobands_time(tf::tock());
      result_mesh->set_polydata(std::move(polys));
      curve_mesh->set_curves_object(std::move(curves));
    }
  }

  auto reset_plane() -> tf::buffer<float> {
    auto points = bridge->get_actors()[0]->poly_object.points();
    auto plane = tf::make_plane(tf::normalized(tf::random_vector<float, 3>()),
                                points[tf::random<int>(0, points.size() - 1)]);
    scalars.allocate(points.size());
    tf::parallel_transform(points, scalars, tf::distance_f(plane));
    distance = 0;
    min_d = *std::min_element(scalars.begin(), scalars.end());
    max_d = *std::max_element(scalars.begin(), scalars.end());
    return scalars;
  }

public:
  auto OnMouseMove(std::array<float, 3>, std::array<float, 3>, std::array<float, 3>, std::array<float, 3>) -> bool override {
    return false;
  }

  auto OnMouseWheel(int delta, bool shiftKey) -> bool override {
    if (shiftKey) {
      distance += (delta * 0.05);
      compute_curves();
      return true;
    }
    return false;
  }

  auto OnKeyPress(std::string key) -> bool override  {
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
  std::cout << "Reading file: " << path << std::endl;
  auto poly = tf::read_stl<int>(path);
  std::cout << "run main 0: " << poly.size() << std::endl;
  if (!poly.size()) {
    std::cout << "Failed to read file" << std::endl;
    throw std::runtime_error("Failed to read file");
  }

  interactor = std::make_unique<cursor_interactor_isobands>();

  utils::center_and_scale_p(poly);
  auto actor = std::make_unique<mesh_object>();
  actor->poly_object = std::move(poly);
  interactor->push_back(std::move(actor));

  if (auto *pI = dynamic_cast<cursor_interactor_isobands*>(interactor.get())) {
    pI->reset_plane();
    pI->compute_curves();
  }

  return 0;
}


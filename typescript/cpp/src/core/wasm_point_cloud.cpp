/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */

#include "trueform/ts/core/wasm_point_cloud.hpp"
#include <emscripten/bind.h>
#include <string>

namespace {

template <typename Real>
auto register_wasm_point_cloud_bindings(const std::string &class_name) -> void {
  using pc_t = tf::ts::wasm_point_cloud<Real>;
  emscripten::class_<pc_t>(class_name.c_str())
      .template constructor<>()
      .class_function("create", &pc_t::create)
      .function("points", &pc_t::points)
      .function("number_of_points", &pc_t::number_of_points)
      .function("set_points", &pc_t::set_points)
      .function("vertex_link", &pc_t::vertex_link)
      .function("has_vertex_link", &pc_t::has_vertex_link)
      .function("set_vertex_link", &pc_t::set_vertex_link)
      .function("normals", &pc_t::normals)
      .function("has_normals", &pc_t::has_normals)
      .function("set_normals", &pc_t::set_normals)
      .function("shallow_copy", &pc_t::shallow_copy)
      .function("has_transformation", &pc_t::has_transformation)
      .function("transformation", &pc_t::transformation)
      .function("set_transformation", &pc_t::set_transformation)
      .function("clear_transformation", &pc_t::clear_transformation)
      .function("build_tree", &pc_t::build_tree)
      .function("destroy", &pc_t::destroy)
      .function("is_valid", &pc_t::is_valid)
      // -- Cache state inspectors (diagnostic) --
      .function("is_tree_built", &pc_t::is_tree_built)
      .function("is_tree_fresh", &pc_t::is_tree_fresh);
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_point_cloud) {
  register_wasm_point_cloud_bindings<float>("NativeFloat32PointCloud");
  register_wasm_point_cloud_bindings<double>("NativeFloat64PointCloud");
}

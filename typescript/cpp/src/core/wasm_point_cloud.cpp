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

namespace tf {
namespace ts {

void wasm_point_cloud::ensure_tree() {
  if (_tree && _tree_gen == _points_gen)
    return;
  _tree = std::make_shared<tf::aabb_tree<int, float, 3>>(
      points_range(), tf::config_tree(4, 4));
  _tree_gen = _points_gen;
}

} // namespace ts
} // namespace tf

EMSCRIPTEN_BINDINGS(trueform_point_cloud) {
  emscripten::class_<tf::ts::wasm_point_cloud>("NativePointCloud")
      .class_function("create", &tf::ts::wasm_point_cloud::create)
      .function("points", &tf::ts::wasm_point_cloud::points)
      .function("number_of_points", &tf::ts::wasm_point_cloud::number_of_points)
      .function("set_points", &tf::ts::wasm_point_cloud::set_points)
      .function("vertex_link", &tf::ts::wasm_point_cloud::vertex_link)
      .function("has_vertex_link", &tf::ts::wasm_point_cloud::has_vertex_link)
      .function("set_vertex_link", &tf::ts::wasm_point_cloud::set_vertex_link)
      .function("normals", &tf::ts::wasm_point_cloud::normals)
      .function("has_normals", &tf::ts::wasm_point_cloud::has_normals)
      .function("set_normals", &tf::ts::wasm_point_cloud::set_normals)
      .function("shared_view", &tf::ts::wasm_point_cloud::shared_view)
      .function("has_transformation", &tf::ts::wasm_point_cloud::has_transformation)
      .function("transformation", &tf::ts::wasm_point_cloud::transformation)
      .function("set_transformation", &tf::ts::wasm_point_cloud::set_transformation)
      .function("clear_transformation", &tf::ts::wasm_point_cloud::clear_transformation)
      .function("ensure_tree", &tf::ts::wasm_point_cloud::ensure_tree)
      .function("destroy", &tf::ts::wasm_point_cloud::destroy)
      .function("is_valid", &tf::ts::wasm_point_cloud::is_valid);
}

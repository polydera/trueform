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
      .function("shallow_copy", &tf::ts::wasm_point_cloud::shallow_copy)
      .function("has_transformation", &tf::ts::wasm_point_cloud::has_transformation)
      .function("transformation", &tf::ts::wasm_point_cloud::transformation)
      .function("set_transformation", &tf::ts::wasm_point_cloud::set_transformation)
      .function("clear_transformation", &tf::ts::wasm_point_cloud::clear_transformation)
      .function("build_tree", &tf::ts::wasm_point_cloud::build_tree)
      .function("destroy", &tf::ts::wasm_point_cloud::destroy)
      .function("is_valid", &tf::ts::wasm_point_cloud::is_valid)
      // -- Cache state inspectors (diagnostic) --
      .function("is_tree_built", &tf::ts::wasm_point_cloud::is_tree_built)
      .function("is_tree_fresh", &tf::ts::wasm_point_cloud::is_tree_fresh);
}

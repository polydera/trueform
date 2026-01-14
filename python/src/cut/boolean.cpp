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

#include <nanobind/nanobind.h>

namespace tf::py {

// Forward declarations for boolean bindings split across multiple files
auto register_boolean_intint33float3d(nanobind::module_ &m) -> void;
auto register_boolean_intint33double3d(nanobind::module_ &m) -> void;
auto register_boolean_intint6433float3d(nanobind::module_ &m) -> void;
auto register_boolean_intint6433double3d(nanobind::module_ &m) -> void;
auto register_boolean_int64int6433float3d(nanobind::module_ &m) -> void;
auto register_boolean_int64int6433double3d(nanobind::module_ &m) -> void;

auto register_cut_boolean(nanobind::module_ &m) -> void {
  // Register all boolean operation bindings
  // Split across multiple files for parallel compilation
  register_boolean_intint33float3d(m);
  register_boolean_intint33double3d(m);
  register_boolean_intint6433float3d(m);
  register_boolean_intint6433double3d(m);
  register_boolean_int64int6433float3d(m);
  register_boolean_int64int6433double3d(m);
}

} // namespace tf::py

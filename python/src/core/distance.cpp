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

// Forward declarations for distance bindings split across multiple files
auto register_core_distance_float2d(nanobind::module_ &m) -> void;
auto register_core_distance_float3d(nanobind::module_ &m) -> void;
auto register_core_distance_double2d(nanobind::module_ &m) -> void;
auto register_core_distance_double3d(nanobind::module_ &m) -> void;

auto register_core_distance(nanobind::module_ &m) -> void {
  // Register all distance bindings
  // Split across multiple files for parallel compilation
  register_core_distance_float2d(m);
  register_core_distance_float3d(m);
  register_core_distance_double2d(m);
  register_core_distance_double3d(m);
}

} // namespace tf::py

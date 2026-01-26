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

// Forward declarations for orient_faces_consistently bindings split across multiple files
auto register_orient_faces_consistently_float2d(nanobind::module_ &m) -> void;
auto register_orient_faces_consistently_float3d(nanobind::module_ &m) -> void;
auto register_orient_faces_consistently_double2d(nanobind::module_ &m) -> void;
auto register_orient_faces_consistently_double3d(nanobind::module_ &m) -> void;

auto register_topology_orient_faces_consistently(nanobind::module_ &m) -> void {
  // Register all orient_faces_consistently bindings
  // Split across multiple files for parallel compilation
  register_orient_faces_consistently_float2d(m);
  register_orient_faces_consistently_float3d(m);
  register_orient_faces_consistently_double2d(m);
  register_orient_faces_consistently_double3d(m);
}

} // namespace tf::py

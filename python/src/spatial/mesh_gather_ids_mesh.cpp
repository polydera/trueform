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

// Forward declarations for mesh_gather_ids_mesh bindings split across multiple files
auto register_mesh_gather_ids_mesh_float2d(nanobind::module_ &m) -> void;
auto register_mesh_gather_ids_mesh_float3d(nanobind::module_ &m) -> void;
auto register_mesh_gather_ids_mesh_double2d(nanobind::module_ &m) -> void;
auto register_mesh_gather_ids_mesh_double3d(nanobind::module_ &m) -> void;

auto register_mesh_gather_ids_mesh(nanobind::module_ &m) -> void {
  // Register all mesh_gather_ids_mesh bindings
  // Split across multiple files for parallel compilation
  register_mesh_gather_ids_mesh_float2d(m);
  register_mesh_gather_ids_mesh_float3d(m);
  register_mesh_gather_ids_mesh_double2d(m);
  register_mesh_gather_ids_mesh_double3d(m);
}

} // namespace tf::py

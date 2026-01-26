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

// Forward declarations for mesh neighbor search bindings split across multiple
// files
auto register_mesh_neighbor_search_int3float2d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_int3float3d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_intdynfloat2d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_intdynfloat3d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_int3double2d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_int3double3d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_intdyndouble2d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_intdyndouble3d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_int643float2d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_int643float3d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_int64dynfloat2d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_int64dynfloat3d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_int643double2d(nanobind::module_ &m)
    -> void;
auto register_mesh_neighbor_search_int643double3d(nanobind::module_ &m)
    -> void;
auto register_mesh_neighbor_search_int64dyndouble2d(nanobind::module_ &m)
    -> void;
auto register_mesh_neighbor_search_int64dyndouble3d(nanobind::module_ &m)
    -> void;

auto register_mesh_neighbor_search(nanobind::module_ &m) -> void {
  // Register all mesh neighbor search bindings
  // Split across multiple files for parallel compilation
  register_mesh_neighbor_search_int3float2d(m);
  register_mesh_neighbor_search_int3float3d(m);
  register_mesh_neighbor_search_intdynfloat2d(m);
  register_mesh_neighbor_search_intdynfloat3d(m);
  register_mesh_neighbor_search_int3double2d(m);
  register_mesh_neighbor_search_int3double3d(m);
  register_mesh_neighbor_search_intdyndouble2d(m);
  register_mesh_neighbor_search_intdyndouble3d(m);
  register_mesh_neighbor_search_int643float2d(m);
  register_mesh_neighbor_search_int643float3d(m);
  register_mesh_neighbor_search_int64dynfloat2d(m);
  register_mesh_neighbor_search_int64dynfloat3d(m);
  register_mesh_neighbor_search_int643double2d(m);
  register_mesh_neighbor_search_int643double3d(m);
  register_mesh_neighbor_search_int64dyndouble2d(m);
  register_mesh_neighbor_search_int64dyndouble3d(m);
}

} // namespace tf::py

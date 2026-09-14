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

// Split across multiple files for parallel compilation
auto register_volume_make_mesh_sdf_int3float3d(nanobind::module_ &m) -> void;
auto register_volume_make_mesh_sdf_int3double3d(nanobind::module_ &m) -> void;
auto register_volume_make_mesh_sdf_int643float3d(nanobind::module_ &m) -> void;
auto register_volume_make_mesh_sdf_int643double3d(nanobind::module_ &m) -> void;
auto register_volume_make_mesh_sdf_intdynfloat3d(nanobind::module_ &m) -> void;
auto register_volume_make_mesh_sdf_intdyndouble3d(nanobind::module_ &m) -> void;
auto register_volume_make_mesh_sdf_int64dynfloat3d(nanobind::module_ &m)
    -> void;
auto register_volume_make_mesh_sdf_int64dyndouble3d(nanobind::module_ &m)
    -> void;

auto register_volume_make_mesh_sdf(nanobind::module_ &m) -> void {
  register_volume_make_mesh_sdf_int3float3d(m);
  register_volume_make_mesh_sdf_int3double3d(m);
  register_volume_make_mesh_sdf_int643float3d(m);
  register_volume_make_mesh_sdf_int643double3d(m);
  register_volume_make_mesh_sdf_intdynfloat3d(m);
  register_volume_make_mesh_sdf_intdyndouble3d(m);
  register_volume_make_mesh_sdf_int64dynfloat3d(m);
  register_volume_make_mesh_sdf_int64dyndouble3d(m);
}

} // namespace tf::py

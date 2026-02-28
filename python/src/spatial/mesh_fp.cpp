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
 * Author: Ziga Sajovic
 */

#include <nanobind/nanobind.h>

namespace tf::py {

auto register_mesh_fp_int3float2d(nanobind::module_ &m) -> void;
auto register_mesh_fp_int3float3d(nanobind::module_ &m) -> void;
auto register_mesh_fp_intdynfloat2d(nanobind::module_ &m) -> void;
auto register_mesh_fp_intdynfloat3d(nanobind::module_ &m) -> void;
auto register_mesh_fp_int3double2d(nanobind::module_ &m) -> void;
auto register_mesh_fp_int3double3d(nanobind::module_ &m) -> void;
auto register_mesh_fp_intdyndouble2d(nanobind::module_ &m) -> void;
auto register_mesh_fp_intdyndouble3d(nanobind::module_ &m) -> void;
auto register_mesh_fp_int643float2d(nanobind::module_ &m) -> void;
auto register_mesh_fp_int643float3d(nanobind::module_ &m) -> void;
auto register_mesh_fp_int64dynfloat2d(nanobind::module_ &m) -> void;
auto register_mesh_fp_int64dynfloat3d(nanobind::module_ &m) -> void;
auto register_mesh_fp_int643double2d(nanobind::module_ &m) -> void;
auto register_mesh_fp_int643double3d(nanobind::module_ &m) -> void;
auto register_mesh_fp_int64dyndouble2d(nanobind::module_ &m) -> void;
auto register_mesh_fp_int64dyndouble3d(nanobind::module_ &m) -> void;

auto register_mesh_fp(nanobind::module_ &m) -> void {
  register_mesh_fp_int3float2d(m);
  register_mesh_fp_int3float3d(m);
  register_mesh_fp_intdynfloat2d(m);
  register_mesh_fp_intdynfloat3d(m);
  register_mesh_fp_int3double2d(m);
  register_mesh_fp_int3double3d(m);
  register_mesh_fp_intdyndouble2d(m);
  register_mesh_fp_intdyndouble3d(m);
  register_mesh_fp_int643float2d(m);
  register_mesh_fp_int643float3d(m);
  register_mesh_fp_int64dynfloat2d(m);
  register_mesh_fp_int64dynfloat3d(m);
  register_mesh_fp_int643double2d(m);
  register_mesh_fp_int643double3d(m);
  register_mesh_fp_int64dyndouble2d(m);
  register_mesh_fp_int64dyndouble3d(m);
}

} // namespace tf::py

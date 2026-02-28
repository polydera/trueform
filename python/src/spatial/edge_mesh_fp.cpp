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

auto register_edge_mesh_fp_intfloat2d(nanobind::module_ &m) -> void;
auto register_edge_mesh_fp_intfloat3d(nanobind::module_ &m) -> void;
auto register_edge_mesh_fp_intdouble2d(nanobind::module_ &m) -> void;
auto register_edge_mesh_fp_intdouble3d(nanobind::module_ &m) -> void;
auto register_edge_mesh_fp_int64float2d(nanobind::module_ &m) -> void;
auto register_edge_mesh_fp_int64float3d(nanobind::module_ &m) -> void;
auto register_edge_mesh_fp_int64double2d(nanobind::module_ &m) -> void;
auto register_edge_mesh_fp_int64double3d(nanobind::module_ &m) -> void;

auto register_edge_mesh_fp(nanobind::module_ &m) -> void {
  register_edge_mesh_fp_intfloat2d(m);
  register_edge_mesh_fp_intfloat3d(m);
  register_edge_mesh_fp_intdouble2d(m);
  register_edge_mesh_fp_intdouble3d(m);
  register_edge_mesh_fp_int64float2d(m);
  register_edge_mesh_fp_int64float3d(m);
  register_edge_mesh_fp_int64double2d(m);
  register_edge_mesh_fp_int64double3d(m);
}

} // namespace tf::py

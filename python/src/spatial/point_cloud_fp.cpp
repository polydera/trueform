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

auto register_point_cloud_fp_float2d(nanobind::module_ &m) -> void;
auto register_point_cloud_fp_float3d(nanobind::module_ &m) -> void;
auto register_point_cloud_fp_double2d(nanobind::module_ &m) -> void;
auto register_point_cloud_fp_double3d(nanobind::module_ &m) -> void;

auto register_point_cloud_fp(nanobind::module_ &m) -> void {
  register_point_cloud_fp_float2d(m);
  register_point_cloud_fp_float3d(m);
  register_point_cloud_fp_double2d(m);
  register_point_cloud_fp_double3d(m);
}

} // namespace tf::py

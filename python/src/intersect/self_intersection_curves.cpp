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

// Forward declarations for self_intersection_curves bindings split across multiple files
auto register_self_intersection_curves_int3float3d(nanobind::module_ &m) -> void;
auto register_self_intersection_curves_int3double3d(nanobind::module_ &m) -> void;
auto register_self_intersection_curves_intdynfloat3d(nanobind::module_ &m) -> void;
auto register_self_intersection_curves_intdyndouble3d(nanobind::module_ &m) -> void;
auto register_self_intersection_curves_int643float3d(nanobind::module_ &m) -> void;
auto register_self_intersection_curves_int643double3d(nanobind::module_ &m) -> void;
auto register_self_intersection_curves_int64dynfloat3d(nanobind::module_ &m) -> void;
auto register_self_intersection_curves_int64dyndouble3d(nanobind::module_ &m) -> void;

auto register_intersect_self_intersection_curves(nanobind::module_ &m) -> void {
  // Register all self_intersection_curves bindings
  // Split across multiple files for parallel compilation
  register_self_intersection_curves_int3float3d(m);
  register_self_intersection_curves_int3double3d(m);
  register_self_intersection_curves_intdynfloat3d(m);
  register_self_intersection_curves_intdyndouble3d(m);
  register_self_intersection_curves_int643float3d(m);
  register_self_intersection_curves_int643double3d(m);
  register_self_intersection_curves_int64dynfloat3d(m);
  register_self_intersection_curves_int64dyndouble3d(m);
}

} // namespace tf::py

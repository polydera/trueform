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

// Forward declarations for has_self_intersections bindings split across
// multiple files
auto register_has_self_intersections_int3float3d(nanobind::module_ &m) -> void;
auto register_has_self_intersections_int3double3d(nanobind::module_ &m) -> void;
auto register_has_self_intersections_intdynfloat3d(nanobind::module_ &m) -> void;
auto register_has_self_intersections_intdyndouble3d(nanobind::module_ &m) -> void;
auto register_has_self_intersections_int643float3d(nanobind::module_ &m) -> void;
auto register_has_self_intersections_int643double3d(nanobind::module_ &m) -> void;
auto register_has_self_intersections_int64dynfloat3d(nanobind::module_ &m) -> void;
auto register_has_self_intersections_int64dyndouble3d(nanobind::module_ &m) -> void;

auto register_intersect_has_self_intersections(nanobind::module_ &m) -> void {
  // Register all has_self_intersections bindings
  // Split across multiple files for parallel compilation
  register_has_self_intersections_int3float3d(m);
  register_has_self_intersections_int3double3d(m);
  register_has_self_intersections_intdynfloat3d(m);
  register_has_self_intersections_intdyndouble3d(m);
  register_has_self_intersections_int643float3d(m);
  register_has_self_intersections_int643double3d(m);
  register_has_self_intersections_int64dynfloat3d(m);
  register_has_self_intersections_int64dyndouble3d(m);
}

} // namespace tf::py

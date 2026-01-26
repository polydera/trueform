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

// Forward declarations for intersection_curves bindings split across multiple files
auto register_intersection_curves_intint_float3d(nanobind::module_ &m) -> void;
auto register_intersection_curves_intint_double3d(nanobind::module_ &m) -> void;
auto register_intersection_curves_intint64_float3d(nanobind::module_ &m) -> void;
auto register_intersection_curves_intint64_double3d(nanobind::module_ &m) -> void;
auto register_intersection_curves_int64int64_float3d(nanobind::module_ &m) -> void;
auto register_intersection_curves_int64int64_double3d(nanobind::module_ &m) -> void;

auto register_intersect_intersection_curves(nanobind::module_ &m) -> void {
  // Register all intersection_curves bindings
  // Split across multiple files for parallel compilation
  register_intersection_curves_intint_float3d(m);
  register_intersection_curves_intint_double3d(m);
  register_intersection_curves_intint64_float3d(m);
  register_intersection_curves_intint64_double3d(m);
  register_intersection_curves_int64int64_float3d(m);
  register_intersection_curves_int64int64_double3d(m);
}

} // namespace tf::py

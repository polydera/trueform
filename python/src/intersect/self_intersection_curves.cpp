/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include <nanobind/nanobind.h>

namespace tf::py {

// Forward declarations for self_intersection_curves bindings split across multiple files
auto register_self_intersection_curves_int3float3d(nanobind::module_ &m) -> void;
auto register_self_intersection_curves_int3double3d(nanobind::module_ &m) -> void;
auto register_self_intersection_curves_int4float3d(nanobind::module_ &m) -> void;
auto register_self_intersection_curves_int4double3d(nanobind::module_ &m) -> void;
auto register_self_intersection_curves_int643float3d(nanobind::module_ &m) -> void;
auto register_self_intersection_curves_int643double3d(nanobind::module_ &m) -> void;
auto register_self_intersection_curves_int644float3d(nanobind::module_ &m) -> void;
auto register_self_intersection_curves_int644double3d(nanobind::module_ &m) -> void;

auto register_intersect_self_intersection_curves(nanobind::module_ &m) -> void {
  // Register all self_intersection_curves bindings
  // Split across multiple files for parallel compilation
  register_self_intersection_curves_int3float3d(m);
  register_self_intersection_curves_int3double3d(m);
  register_self_intersection_curves_int4float3d(m);
  register_self_intersection_curves_int4double3d(m);
  register_self_intersection_curves_int643float3d(m);
  register_self_intersection_curves_int643double3d(m);
  register_self_intersection_curves_int644float3d(m);
  register_self_intersection_curves_int644double3d(m);
}

} // namespace tf::py

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

// Forward declarations for embedded_self_intersection_curves bindings split across multiple files
// Note: Only triangles (ngon=3) supported, no quads
auto register_embedded_self_intersection_curves_int3float3d(nanobind::module_ &m) -> void;
auto register_embedded_self_intersection_curves_int3double3d(nanobind::module_ &m) -> void;
auto register_embedded_self_intersection_curves_int643float3d(nanobind::module_ &m) -> void;
auto register_embedded_self_intersection_curves_int643double3d(nanobind::module_ &m) -> void;

auto register_cut_embedded_self_intersection_curves(nanobind::module_ &m) -> void {
  // Register all embedded_self_intersection_curves bindings
  // Split across multiple files for parallel compilation
  // Note: Only triangles (ngon=3) supported
  register_embedded_self_intersection_curves_int3float3d(m);
  register_embedded_self_intersection_curves_int3double3d(m);
  register_embedded_self_intersection_curves_int643float3d(m);
  register_embedded_self_intersection_curves_int643double3d(m);
}

} // namespace tf::py

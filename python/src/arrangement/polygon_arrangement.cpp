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

auto register_polygon_arrangement_int3float3d(nanobind::module_ &m) -> void;
auto register_polygon_arrangement_int3double3d(nanobind::module_ &m) -> void;
auto register_polygon_arrangement_int643float3d(nanobind::module_ &m) -> void;
auto register_polygon_arrangement_int643double3d(nanobind::module_ &m) -> void;

auto register_arrangement_polygon_arrangement(nanobind::module_ &m) -> void {
  register_polygon_arrangement_int3float3d(m);
  register_polygon_arrangement_int3double3d(m);
  register_polygon_arrangement_int643float3d(m);
  register_polygon_arrangement_int643double3d(m);
}

} // namespace tf::py

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

auto register_embedded_intersection_curves_intint33float3d(nanobind::module_ &m) -> void;
auto register_embedded_intersection_curves_intint33double3d(nanobind::module_ &m) -> void;
auto register_embedded_intersection_curves_intint6433float3d(nanobind::module_ &m) -> void;
auto register_embedded_intersection_curves_intint6433double3d(nanobind::module_ &m) -> void;
auto register_embedded_intersection_curves_int64int33float3d(nanobind::module_ &m) -> void;
auto register_embedded_intersection_curves_int64int33double3d(nanobind::module_ &m) -> void;
auto register_embedded_intersection_curves_int64int6433float3d(nanobind::module_ &m) -> void;
auto register_embedded_intersection_curves_int64int6433double3d(nanobind::module_ &m) -> void;

auto register_cut_embedded_intersection_curves(nanobind::module_ &m) -> void {
  register_embedded_intersection_curves_intint33float3d(m);
  register_embedded_intersection_curves_intint33double3d(m);
  register_embedded_intersection_curves_intint6433float3d(m);
  register_embedded_intersection_curves_intint6433double3d(m);
  register_embedded_intersection_curves_int64int33float3d(m);
  register_embedded_intersection_curves_int64int33double3d(m);
  register_embedded_intersection_curves_int64int6433float3d(m);
  register_embedded_intersection_curves_int64int6433double3d(m);
}

} // namespace tf::py

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

#include "trueform/python/cut/embedded_self_intersection_curves.hpp"

namespace tf::py {

auto register_embedded_self_intersection_curves_int3float3d(nanobind::module_ &m) -> void {
  // int32, triangles, float32, 3D

  // Without curves
  m.def("embedded_self_intersection_curves_mesh_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh) {
          return embedded_self_intersection_curves(mesh);
        },
        nanobind::arg("mesh"));

  // With curves
  m.def("embedded_self_intersection_curves_curves_mesh_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh) {
          return embedded_self_intersection_curves(mesh, tf::return_curves);
        },
        nanobind::arg("mesh"));

  // int32, dynamic, float32, 3D

  // Without curves
  m.def("embedded_self_intersection_curves_mesh_intdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh) {
          return embedded_self_intersection_curves(mesh);
        },
        nanobind::arg("mesh"));

  // With curves
  m.def("embedded_self_intersection_curves_curves_mesh_intdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh) {
          return embedded_self_intersection_curves(mesh, tf::return_curves);
        },
        nanobind::arg("mesh"));
}

} // namespace tf::py

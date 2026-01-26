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

auto register_embedded_self_intersection_curves_int643double3d(nanobind::module_ &m) -> void {
  // int64, triangles, float64, 3D

  // Without curves
  m.def("embedded_self_intersection_curves_mesh_int643double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh) {
          return embedded_self_intersection_curves(mesh);
        },
        nanobind::arg("mesh"));

  // With curves
  m.def("embedded_self_intersection_curves_curves_mesh_int643double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh) {
          return embedded_self_intersection_curves(mesh, tf::return_curves);
        },
        nanobind::arg("mesh"));

  // int64, dynamic, float64, 3D

  // Without curves
  m.def("embedded_self_intersection_curves_mesh_int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh) {
          return embedded_self_intersection_curves(mesh);
        },
        nanobind::arg("mesh"));

  // With curves
  m.def("embedded_self_intersection_curves_curves_mesh_int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh) {
          return embedded_self_intersection_curves(mesh, tf::return_curves);
        },
        nanobind::arg("mesh"));
}

} // namespace tf::py

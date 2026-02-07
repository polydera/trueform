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

#include "trueform/python/cut/embedded_intersection_curves.hpp"

namespace tf::py {

auto register_embedded_intersection_curves_int64int33double3d(nanobind::module_ &m) -> void {
  // int64 × int32, float64, triangles, 3D

  // 3×3 without curves
  m.def("embedded_intersection_curves_mesh_mesh_int64int33double3d",
        [](mesh_wrapper<std::int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1) {
          return embedded_intersection_curves(mesh0, mesh1);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"));

  // 3×3 with curves
  m.def("embedded_intersection_curves_curves_mesh_mesh_int64int33double3d",
        [](mesh_wrapper<std::int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1) {
          return embedded_intersection_curves(mesh0, mesh1, tf::return_curves);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"));

  // 3×dyn without curves
  m.def("embedded_intersection_curves_mesh_mesh_int64int3dyndouble3d",
        [](mesh_wrapper<std::int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, dynamic_size, 3> &mesh1) {
          return embedded_intersection_curves(mesh0, mesh1);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"));

  // 3×dyn with curves
  m.def("embedded_intersection_curves_curves_mesh_mesh_int64int3dyndouble3d",
        [](mesh_wrapper<std::int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, dynamic_size, 3> &mesh1) {
          return embedded_intersection_curves(mesh0, mesh1, tf::return_curves);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"));

  // dyn×3 without curves
  m.def("embedded_intersection_curves_mesh_mesh_int64intdyn3double3d",
        [](mesh_wrapper<std::int64_t, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1) {
          return embedded_intersection_curves(mesh0, mesh1);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"));

  // dyn×3 with curves
  m.def("embedded_intersection_curves_curves_mesh_mesh_int64intdyn3double3d",
        [](mesh_wrapper<std::int64_t, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1) {
          return embedded_intersection_curves(mesh0, mesh1, tf::return_curves);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"));

  // dyn×dyn without curves
  m.def("embedded_intersection_curves_mesh_mesh_int64intdyndyndouble3d",
        [](mesh_wrapper<std::int64_t, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int, double, dynamic_size, 3> &mesh1) {
          return embedded_intersection_curves(mesh0, mesh1);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"));

  // dyn×dyn with curves
  m.def("embedded_intersection_curves_curves_mesh_mesh_int64intdyndyndouble3d",
        [](mesh_wrapper<std::int64_t, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int, double, dynamic_size, 3> &mesh1) {
          return embedded_intersection_curves(mesh0, mesh1, tf::return_curves);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"));
}

} // namespace tf::py

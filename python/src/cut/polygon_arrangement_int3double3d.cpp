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

#include "trueform/python/cut/polygon_arrangement_impl.hpp"

namespace tf::py {

auto register_polygon_arrangement_int3double3d(nanobind::module_ &m) -> void {
  m.def("polygon_arrangements_int3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh, int mode, double tolerance, int triangulation) {
          return polygon_arrangements(mesh, mode, tolerance, triangulation);
        },
        nanobind::arg("mesh"), nanobind::arg("mode"), nanobind::arg("tolerance") = 0.0, nanobind::arg("triangulation") = 0);

  m.def("polygon_arrangements_curves_int3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh, int mode, double tolerance, int triangulation) {
          return polygon_arrangements(mesh, mode, tolerance, triangulation, tf::return_curves);
        },
        nanobind::arg("mesh"), nanobind::arg("mode"), nanobind::arg("tolerance") = 0.0, nanobind::arg("triangulation") = 0);

  m.def("polygon_arrangements_intdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh, int mode, double tolerance, int triangulation) {
          return polygon_arrangements(mesh, mode, tolerance, triangulation);
        },
        nanobind::arg("mesh"), nanobind::arg("mode"), nanobind::arg("tolerance") = 0.0, nanobind::arg("triangulation") = 0);

  m.def("polygon_arrangements_curves_intdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh, int mode, double tolerance, int triangulation) {
          return polygon_arrangements(mesh, mode, tolerance, triangulation, tf::return_curves);
        },
        nanobind::arg("mesh"), nanobind::arg("mode"), nanobind::arg("tolerance") = 0.0, nanobind::arg("triangulation") = 0);
}

} // namespace tf::py

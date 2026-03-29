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
        [](mesh_wrapper<int, double, 3, 3> &mesh) {
          return polygon_arrangements(mesh);
        },
        nanobind::arg("mesh"));

  m.def("polygon_arrangements_curves_int3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh) {
          return polygon_arrangements(mesh, tf::return_curves);
        },
        nanobind::arg("mesh"));

  m.def("polygon_arrangements_intdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh) {
          return polygon_arrangements(mesh);
        },
        nanobind::arg("mesh"));

  m.def("polygon_arrangements_curves_intdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh) {
          return polygon_arrangements(mesh, tf::return_curves);
        },
        nanobind::arg("mesh"));
}

} // namespace tf::py

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

#include "trueform/python/arrangement/mesh_arrangement_impl.hpp"

namespace tf::py {

auto register_mesh_arrangements_int643double3d(nanobind::module_ &m) -> void {
  using W = mesh_wrapper<int64_t, double, 3, 3>;

  m.def("mesh_arrangements_int643double3d",
        [](std::vector<W> &meshes, int mode, double tolerance, int triangulation) { return mesh_arrangements(meshes, mode, tolerance, triangulation); },
        nanobind::arg("meshes"), nanobind::arg("mode"), nanobind::arg("tolerance") = 0.0, nanobind::arg("triangulation") = 0);

  m.def("mesh_arrangements_curves_int643double3d",
        [](std::vector<W> &meshes, int mode, double tolerance, int triangulation) {
          return mesh_arrangements(meshes, mode, tolerance, triangulation, tf::return_curves);
        },
        nanobind::arg("meshes"), nanobind::arg("mode"), nanobind::arg("tolerance") = 0.0, nanobind::arg("triangulation") = 0);
}

} // namespace tf::py

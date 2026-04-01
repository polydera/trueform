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

#include "trueform/python/cut/mesh_arrangement_impl.hpp"

namespace tf::py {

auto register_mesh_arrangements_intdynfloat3d(nanobind::module_ &m) -> void {
  using W = mesh_wrapper<int, float, tf::dynamic_size, 3>;

  m.def("mesh_arrangements_intdynfloat3d",
        [](std::vector<W> &meshes, int mode) { return mesh_arrangements(meshes, mode); },
        nanobind::arg("meshes"), nanobind::arg("mode"));

  m.def("mesh_arrangements_curves_intdynfloat3d",
        [](std::vector<W> &meshes, int mode) {
          return mesh_arrangements(meshes, mode, tf::return_curves);
        },
        nanobind::arg("meshes"), nanobind::arg("mode"));
}

} // namespace tf::py

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

auto register_mesh_arrangements_int3double3d(nanobind::module_ &m) -> void {
  using W = mesh_wrapper<int, double, 3, 3>;

  m.def("mesh_arrangements_int3double3d",
        [](std::vector<W> &meshes) { return mesh_arrangements(meshes); },
        nanobind::arg("meshes"));

  m.def("mesh_arrangements_curves_int3double3d",
        [](std::vector<W> &meshes) {
          return mesh_arrangements(meshes, tf::return_curves);
        },
        nanobind::arg("meshes"));
}

} // namespace tf::py

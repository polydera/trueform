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

#include "trueform/python/csg/outer_shell_impl.hpp"

namespace tf::py {

auto register_outer_shell_int3float3d(nanobind::module_ &m) -> void {
  m.def("outer_shell_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh) { return outer_shell(mesh); },
        nanobind::arg("mesh"));

  m.def("outer_shell_intdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh) {
          return outer_shell(mesh);
        },
        nanobind::arg("mesh"));
}

} // namespace tf::py

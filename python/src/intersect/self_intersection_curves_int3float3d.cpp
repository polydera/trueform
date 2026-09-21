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

#include "trueform/python/intersect/self_intersection_curves.hpp"
#include <trueform/intersect/intersect_mode.hpp>

namespace tf::py {

auto register_self_intersection_curves_int3float3d(nanobind::module_ &m) -> void {
  // int32, triangles, float32, 3D
  m.def("self_intersection_curves_mesh_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh, int mode, double tolerance) {
          return self_intersection_curves(
              mesh, static_cast<tf::intersect_mode>(mode), tolerance);
        },
        nanobind::arg("mesh"),
        nanobind::arg("mode") = static_cast<int>(tf::intersect_mode::primitives),
        nanobind::arg("tolerance") = 0.0);
}

} // namespace tf::py

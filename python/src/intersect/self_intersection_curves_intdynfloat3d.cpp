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

namespace tf::py {

auto register_self_intersection_curves_intdynfloat3d(nanobind::module_ &m) -> void {
  // int32, dynamic, float32, 3D
  m.def("self_intersection_curves_mesh_intdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh, int mode, double tolerance) {
          return self_intersection_curves(
              mesh, static_cast<tf::intersect_mode>(mode), tolerance);
        },
        nanobind::arg("mesh"), nanobind::arg("mode") = 0, nanobind::arg("tolerance") = 0.0);
}

} // namespace tf::py

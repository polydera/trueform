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

auto register_self_intersection_curves_int64dyndouble3d(nanobind::module_ &m) -> void {
  // int64, dynamic, float64, 3D
  m.def("self_intersection_curves_mesh_int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh, int mode) {
          return self_intersection_curves(
              mesh, static_cast<tf::intersect_mode>(mode));
        },
        nanobind::arg("mesh"), nanobind::arg("mode") = 0);
}

} // namespace tf::py

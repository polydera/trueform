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
#include "trueform/python/intersect/intersection_curves.hpp"

namespace tf::py {

auto register_intersection_curves_intint64_double3d(nanobind::module_ &m) -> void {
  // int32 × int64, double, 3D

  m.def("intersection_curves_mesh_mesh_intint6433double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint643dyndouble3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint64dyn3double3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint64dyndyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });
}

} // namespace tf::py

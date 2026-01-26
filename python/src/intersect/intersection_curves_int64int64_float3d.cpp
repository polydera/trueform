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

auto register_intersection_curves_int64int64_float3d(nanobind::module_ &m) -> void {
  // int64 × int64, float, 3D

  m.def("intersection_curves_mesh_mesh_int64int6433float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_int64int643dynfloat3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_int64int64dyn3float3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_int64int64dyndynfloat3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });
}

} // namespace tf::py

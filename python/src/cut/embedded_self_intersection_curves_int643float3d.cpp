/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/cut/embedded_self_intersection_curves.hpp"

namespace tf::py {

auto register_embedded_self_intersection_curves_int643float3d(nanobind::module_ &m) -> void {
  // int64, triangles, float32, 3D

  // Without curves
  m.def("embedded_self_intersection_curves_mesh_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh) {
          return embedded_self_intersection_curves(mesh);
        },
        nanobind::arg("mesh"));

  // With curves
  m.def("embedded_self_intersection_curves_curves_mesh_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh) {
          return embedded_self_intersection_curves(mesh, tf::return_curves);
        },
        nanobind::arg("mesh"));
}

} // namespace tf::py

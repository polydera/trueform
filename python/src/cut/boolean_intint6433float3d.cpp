/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/cut/boolean_impl.hpp"

namespace tf::py {

auto register_boolean_intint6433float3d(nanobind::module_ &m) -> void {
  // int32 × int64, float32, triangles, 3D

  // Without curves
  m.def("boolean_mesh_mesh_intint6433float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1, int op) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op));
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"));

  // With curves
  m.def("boolean_curves_mesh_mesh_intint6433float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1, int op) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op),
                         tf::return_curves);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"));
}

} // namespace tf::py

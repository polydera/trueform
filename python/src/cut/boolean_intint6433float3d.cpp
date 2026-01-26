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

  // 3×dyn without curves
  m.def("boolean_mesh_mesh_intint643dynfloat3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh1, int op) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op));
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"));

  // 3×dyn with curves
  m.def("boolean_curves_mesh_mesh_intint643dynfloat3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh1, int op) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op),
                         tf::return_curves);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"));

  // dyn×3 without curves
  m.def("boolean_mesh_mesh_intint64dyn3float3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1, int op) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op));
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"));

  // dyn×3 with curves
  m.def("boolean_curves_mesh_mesh_intint64dyn3float3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1, int op) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op),
                         tf::return_curves);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"));

  // dyn×dyn without curves
  m.def("boolean_mesh_mesh_intint64dyndynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh1, int op) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op));
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"));

  // dyn×dyn with curves
  m.def("boolean_curves_mesh_mesh_intint64dyndynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh1, int op) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op),
                         tf::return_curves);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"));
}

} // namespace tf::py

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

#include "trueform/python/csg/boolean_impl.hpp"

namespace tf::py {

auto register_boolean_int64int6433double3d(nanobind::module_ &m) -> void {
  // int64 × int64, float64, triangles, 3D

  // Without curves
  m.def("boolean_mesh_mesh_int64int6433double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1, int op,
           const std::vector<int> &sheets) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op), sheets);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"), nanobind::arg("sheets"));

  // With curves
  m.def("boolean_curves_mesh_mesh_int64int6433double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1, int op,
           const std::vector<int> &sheets) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op),
                         tf::return_curves, sheets);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"), nanobind::arg("sheets"));

  // 3×dyn without curves
  m.def("boolean_mesh_mesh_int64int643dyndouble3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh1, int op,
           const std::vector<int> &sheets) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op), sheets);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"), nanobind::arg("sheets"));

  // 3×dyn with curves
  m.def("boolean_curves_mesh_mesh_int64int643dyndouble3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh1, int op,
           const std::vector<int> &sheets) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op),
                         tf::return_curves, sheets);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"), nanobind::arg("sheets"));

  // dyn×3 without curves
  m.def("boolean_mesh_mesh_int64int64dyn3double3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1, int op,
           const std::vector<int> &sheets) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op), sheets);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"), nanobind::arg("sheets"));

  // dyn×3 with curves
  m.def("boolean_curves_mesh_mesh_int64int64dyn3double3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1, int op,
           const std::vector<int> &sheets) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op),
                         tf::return_curves, sheets);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"), nanobind::arg("sheets"));

  // dyn×dyn without curves
  m.def("boolean_mesh_mesh_int64int64dyndyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh1, int op,
           const std::vector<int> &sheets) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op), sheets);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"), nanobind::arg("sheets"));

  // dyn×dyn with curves
  m.def("boolean_curves_mesh_mesh_int64int64dyndyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh1, int op,
           const std::vector<int> &sheets) {
          return boolean(mesh0, mesh1, int_to_boolean_op(op),
                         tf::return_curves, sheets);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"), nanobind::arg("op"), nanobind::arg("sheets"));
}

} // namespace tf::py

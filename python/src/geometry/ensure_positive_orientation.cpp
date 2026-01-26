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
#include <nanobind/nanobind.h>
#include <trueform/python/geometry/ensure_positive_orientation.hpp>
#include <trueform/python/spatial/mesh.hpp>

namespace tf::py {

auto register_ensure_positive_orientation(nanobind::module_ &m) -> void {

  // ==== float, 3D ====

  // int32, ngon=3
  m.def("ensure_positive_orientation_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh, bool is_consistent) {
          return ensure_positive_orientation(mesh, is_consistent);
        },
        nanobind::arg("mesh"), nanobind::arg("is_consistent") = false);

  // int32, dynamic
  m.def("ensure_positive_orientation_intdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh, bool is_consistent) {
          return ensure_positive_orientation(mesh, is_consistent);
        },
        nanobind::arg("mesh"), nanobind::arg("is_consistent") = false);

  // int64, ngon=3
  m.def("ensure_positive_orientation_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh, bool is_consistent) {
          return ensure_positive_orientation(mesh, is_consistent);
        },
        nanobind::arg("mesh"), nanobind::arg("is_consistent") = false);

  // int64, dynamic
  m.def("ensure_positive_orientation_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh, bool is_consistent) {
          return ensure_positive_orientation(mesh, is_consistent);
        },
        nanobind::arg("mesh"), nanobind::arg("is_consistent") = false);

  // ==== double, 3D ====

  // int32, ngon=3
  m.def("ensure_positive_orientation_int3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh, bool is_consistent) {
          return ensure_positive_orientation(mesh, is_consistent);
        },
        nanobind::arg("mesh"), nanobind::arg("is_consistent") = false);

  // int32, dynamic
  m.def("ensure_positive_orientation_intdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh, bool is_consistent) {
          return ensure_positive_orientation(mesh, is_consistent);
        },
        nanobind::arg("mesh"), nanobind::arg("is_consistent") = false);

  // int64, ngon=3
  m.def("ensure_positive_orientation_int643double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh, bool is_consistent) {
          return ensure_positive_orientation(mesh, is_consistent);
        },
        nanobind::arg("mesh"), nanobind::arg("is_consistent") = false);

  // int64, dynamic
  m.def("ensure_positive_orientation_int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh, bool is_consistent) {
          return ensure_positive_orientation(mesh, is_consistent);
        },
        nanobind::arg("mesh"), nanobind::arg("is_consistent") = false);
}

} // namespace tf::py

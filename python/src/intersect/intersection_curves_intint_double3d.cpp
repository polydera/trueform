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

auto register_intersection_curves_intint_double3d(nanobind::module_ &m) -> void {
  // int32 × int32, double, 3D

  m.def("intersection_curves_mesh_mesh_intint33double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1, int mode) {
          return intersection_curves(mesh0, mesh1,
                                     static_cast<tf::intersect_mode>(mode));
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("mode") = 0);

  m.def("intersection_curves_mesh_mesh_intint3dyndouble3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, dynamic_size, 3> &mesh1, int mode) {
          return intersection_curves(mesh0, mesh1,
                                     static_cast<tf::intersect_mode>(mode));
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("mode") = 0);

  m.def("intersection_curves_mesh_mesh_intintdyn3double3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1, int mode) {
          return intersection_curves(mesh0, mesh1,
                                     static_cast<tf::intersect_mode>(mode));
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("mode") = 0);

  m.def("intersection_curves_mesh_mesh_intintdyndyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int, double, dynamic_size, 3> &mesh1, int mode) {
          return intersection_curves(mesh0, mesh1,
                                     static_cast<tf::intersect_mode>(mode));
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("mode") = 0);

  // List overloads
  m.def("intersection_curves_list_int3double3d",
        [](std::vector<mesh_wrapper<int, double, 3, 3>> &meshes, int mode) {
          return intersection_curves(meshes,
                                     static_cast<tf::intersect_mode>(mode));
        },
        nanobind::arg("meshes"), nanobind::arg("mode") = 0);

  m.def("intersection_curves_list_intdyndouble3d",
        [](std::vector<mesh_wrapper<int, double, dynamic_size, 3>> &meshes,
           int mode) {
          return intersection_curves(meshes,
                                     static_cast<tf::intersect_mode>(mode));
        },
        nanobind::arg("meshes"), nanobind::arg("mode") = 0);
}

} // namespace tf::py

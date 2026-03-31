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

auto register_intersection_curves_intint_float3d(nanobind::module_ &m) -> void {
  // int32 × int32, float, 3D

  m.def("intersection_curves_mesh_mesh_intint33float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1, int mode) {
          return intersection_curves(mesh0, mesh1,
                                     static_cast<tf::intersect_mode>(mode));
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("mode") = 0);

  m.def("intersection_curves_mesh_mesh_intint3dynfloat3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, dynamic_size, 3> &mesh1, int mode) {
          return intersection_curves(mesh0, mesh1,
                                     static_cast<tf::intersect_mode>(mode));
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("mode") = 0);

  m.def("intersection_curves_mesh_mesh_intintdyn3float3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1, int mode) {
          return intersection_curves(mesh0, mesh1,
                                     static_cast<tf::intersect_mode>(mode));
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("mode") = 0);

  m.def("intersection_curves_mesh_mesh_intintdyndynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh0,
           mesh_wrapper<int, float, dynamic_size, 3> &mesh1, int mode) {
          return intersection_curves(mesh0, mesh1,
                                     static_cast<tf::intersect_mode>(mode));
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("mode") = 0);

  // List overloads
  m.def("intersection_curves_list_int3float3d",
        [](std::vector<mesh_wrapper<int, float, 3, 3>> &meshes, int mode) {
          return intersection_curves(meshes,
                                     static_cast<tf::intersect_mode>(mode));
        },
        nanobind::arg("meshes"), nanobind::arg("mode") = 0);

  m.def("intersection_curves_list_intdynfloat3d",
        [](std::vector<mesh_wrapper<int, float, dynamic_size, 3>> &meshes,
           int mode) {
          return intersection_curves(meshes,
                                     static_cast<tf::intersect_mode>(mode));
        },
        nanobind::arg("meshes"), nanobind::arg("mode") = 0);
}

} // namespace tf::py

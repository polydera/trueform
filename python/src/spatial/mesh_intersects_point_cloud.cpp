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
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/point_cloud.hpp>
#include <trueform/python/spatial/form_intersects_form.hpp>

namespace tf::py {

auto register_mesh_intersects_point_cloud(nanobind::module_ &m) -> void {

  // ============================================================================
  // Mesh intersects PointCloud
  // Mesh: 2 index types × 2 ngons
  // PointCloud: no index type
  // Real types: float, double (must match)
  // Dims: 2D, 3D (must match)
  // Total: 2 × 2 × 2 × 2 = 16 functions
  // ============================================================================

  // int32, float, triangle, 2D
  m.def("intersects_mesh_point_cloud_int3float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           point_cloud_wrapper<float, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int32, float, dynamic, 2D
  m.def("intersects_mesh_point_cloud_intdynfloat2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh,
           point_cloud_wrapper<float, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int32, float, triangle, 3D
  m.def("intersects_mesh_point_cloud_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           point_cloud_wrapper<float, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int32, float, dynamic, 3D
  m.def("intersects_mesh_point_cloud_intdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
           point_cloud_wrapper<float, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int32, double, triangle, 2D
  m.def("intersects_mesh_point_cloud_int3double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           point_cloud_wrapper<double, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int32, double, dynamic, 2D
  m.def("intersects_mesh_point_cloud_intdyndouble2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh,
           point_cloud_wrapper<double, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int32, double, triangle, 3D
  m.def("intersects_mesh_point_cloud_int3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           point_cloud_wrapper<double, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int32, double, dynamic, 3D
  m.def("intersects_mesh_point_cloud_intdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
           point_cloud_wrapper<double, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, float, triangle, 2D
  m.def("intersects_mesh_point_cloud_int643float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           point_cloud_wrapper<float, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, float, dynamic, 2D
  m.def("intersects_mesh_point_cloud_int64dynfloat2d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh,
           point_cloud_wrapper<float, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, float, triangle, 3D
  m.def("intersects_mesh_point_cloud_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           point_cloud_wrapper<float, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, float, dynamic, 3D
  m.def("intersects_mesh_point_cloud_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
           point_cloud_wrapper<float, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, double, triangle, 2D
  m.def("intersects_mesh_point_cloud_int643double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           point_cloud_wrapper<double, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, double, dynamic, 2D
  m.def("intersects_mesh_point_cloud_int64dyndouble2d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh,
           point_cloud_wrapper<double, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, double, triangle, 3D
  m.def("intersects_mesh_point_cloud_int643double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           point_cloud_wrapper<double, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, double, dynamic, 3D
  m.def("intersects_mesh_point_cloud_int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
           point_cloud_wrapper<double, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));
}

} // namespace tf::py

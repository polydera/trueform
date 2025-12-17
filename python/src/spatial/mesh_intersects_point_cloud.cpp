/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
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
  m.def("intersects_mesh_point_cloud_intfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           point_cloud_wrapper<float, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int32, float, dynamic, 2D
  m.def("intersects_mesh_point_cloud_intfloatdyn2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh,
           point_cloud_wrapper<float, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int32, float, triangle, 3D
  m.def("intersects_mesh_point_cloud_intfloat33d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           point_cloud_wrapper<float, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int32, float, dynamic, 3D
  m.def("intersects_mesh_point_cloud_intfloatdyn3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
           point_cloud_wrapper<float, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int32, double, triangle, 2D
  m.def("intersects_mesh_point_cloud_intdouble32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           point_cloud_wrapper<double, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int32, double, dynamic, 2D
  m.def("intersects_mesh_point_cloud_intdoubledyn2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh,
           point_cloud_wrapper<double, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int32, double, triangle, 3D
  m.def("intersects_mesh_point_cloud_intdouble33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           point_cloud_wrapper<double, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int32, double, dynamic, 3D
  m.def("intersects_mesh_point_cloud_intdoubledyn3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
           point_cloud_wrapper<double, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, float, triangle, 2D
  m.def("intersects_mesh_point_cloud_int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           point_cloud_wrapper<float, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, float, dynamic, 2D
  m.def("intersects_mesh_point_cloud_int64floatdyn2d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh,
           point_cloud_wrapper<float, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, float, triangle, 3D
  m.def("intersects_mesh_point_cloud_int64float33d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           point_cloud_wrapper<float, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, float, dynamic, 3D
  m.def("intersects_mesh_point_cloud_int64floatdyn3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
           point_cloud_wrapper<float, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, double, triangle, 2D
  m.def("intersects_mesh_point_cloud_int64double32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           point_cloud_wrapper<double, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, double, dynamic, 2D
  m.def("intersects_mesh_point_cloud_int64doubledyn2d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh,
           point_cloud_wrapper<double, 2> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, double, triangle, 3D
  m.def("intersects_mesh_point_cloud_int64double33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           point_cloud_wrapper<double, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));

  // int64, double, dynamic, 3D
  m.def("intersects_mesh_point_cloud_int64doubledyn3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
           point_cloud_wrapper<double, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));
}

} // namespace tf::py

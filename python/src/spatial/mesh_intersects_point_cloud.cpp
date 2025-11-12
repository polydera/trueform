/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <trueform/python/core/mesh.hpp>
#include <trueform/python/core/point_cloud.hpp>
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

  // int32, float, quad, 2D
  m.def("intersects_mesh_point_cloud_intfloat42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
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

  // int32, float, quad, 3D
  m.def("intersects_mesh_point_cloud_intfloat43d",
        [](mesh_wrapper<int, float, 4, 3> &mesh,
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

  // int32, double, quad, 2D
  m.def("intersects_mesh_point_cloud_intdouble42d",
        [](mesh_wrapper<int, double, 4, 2> &mesh,
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

  // int32, double, quad, 3D
  m.def("intersects_mesh_point_cloud_intdouble43d",
        [](mesh_wrapper<int, double, 4, 3> &mesh,
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

  // int64, float, quad, 2D
  m.def("intersects_mesh_point_cloud_int64float42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
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

  // int64, float, quad, 3D
  m.def("intersects_mesh_point_cloud_int64float43d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
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

  // int64, double, quad, 2D
  m.def("intersects_mesh_point_cloud_int64double42d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
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

  // int64, double, quad, 3D
  m.def("intersects_mesh_point_cloud_int64double43d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
           point_cloud_wrapper<double, 3> &cloud) {
          return form_intersects_form(mesh, cloud);
        },
        nanobind::arg("mesh"), nanobind::arg("cloud"));
}

} // namespace tf::py

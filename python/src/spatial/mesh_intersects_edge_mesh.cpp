/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <trueform/python/spatial/edge_mesh.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/form_intersects_form.hpp>

namespace tf::py {

auto register_mesh_intersects_edge_mesh(nanobind::module_ &m) -> void {

  // ============================================================================
  // Mesh intersects EdgeMesh
  // Mesh: 2 index types × 2 ngons
  // EdgeMesh: 2 index types
  // Real types: float, double (must match)
  // Dims: 2D, 3D (must match)
  // Total: 2 × 2 × 2 × 2 × 2 = 32 functions
  // ============================================================================

  // ==== float, 2D ====

  // int32 mesh, int32 edge_mesh, triangle, float, 2D
  m.def("intersects_mesh_edge_mesh_intintfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, triangle, float, 2D
  m.def("intersects_mesh_edge_mesh_intint64float32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, triangle, float, 2D
  m.def("intersects_mesh_edge_mesh_int64int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int32 edge_mesh, dynamic, float, 2D
  m.def("intersects_mesh_edge_mesh_intintfloatdyn2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, dynamic, float, 2D
  m.def("intersects_mesh_edge_mesh_intint64floatdyn2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, dynamic, float, 2D
  m.def("intersects_mesh_edge_mesh_int64int64floatdyn2d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // ==== float, 3D ====

  // int32 mesh, int32 edge_mesh, triangle, float, 3D
  m.def("intersects_mesh_edge_mesh_intintfloat33d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, triangle, float, 3D
  m.def("intersects_mesh_edge_mesh_intint64float33d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, triangle, float, 3D
  m.def("intersects_mesh_edge_mesh_int64int64float33d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int32 edge_mesh, dynamic, float, 3D
  m.def("intersects_mesh_edge_mesh_intintfloatdyn3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, dynamic, float, 3D
  m.def("intersects_mesh_edge_mesh_intint64floatdyn3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, dynamic, float, 3D
  m.def("intersects_mesh_edge_mesh_int64int64floatdyn3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // ==== double, 2D ====

  // int32 mesh, int32 edge_mesh, triangle, double, 2D
  m.def("intersects_mesh_edge_mesh_intintdouble32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, triangle, double, 2D
  m.def("intersects_mesh_edge_mesh_intint64double32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, triangle, double, 2D
  m.def("intersects_mesh_edge_mesh_int64int64double32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int32 edge_mesh, dynamic, double, 2D
  m.def("intersects_mesh_edge_mesh_intintdoubledyn2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, dynamic, double, 2D
  m.def("intersects_mesh_edge_mesh_intint64doubledyn2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, dynamic, double, 2D
  m.def("intersects_mesh_edge_mesh_int64int64doubledyn2d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // ==== double, 3D ====

  // int32 mesh, int32 edge_mesh, triangle, double, 3D
  m.def("intersects_mesh_edge_mesh_intintdouble33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, triangle, double, 3D
  m.def("intersects_mesh_edge_mesh_intint64double33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, triangle, double, 3D
  m.def("intersects_mesh_edge_mesh_int64int64double33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int32 edge_mesh, dynamic, double, 3D
  m.def("intersects_mesh_edge_mesh_intintdoubledyn3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, dynamic, double, 3D
  m.def("intersects_mesh_edge_mesh_intint64doubledyn3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, dynamic, double, 3D
  m.def("intersects_mesh_edge_mesh_int64int64doubledyn3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, triangle, float, 2D
  m.def("intersects_mesh_edge_mesh_int64intfloat32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, dynamic, float, 2D
  m.def("intersects_mesh_edge_mesh_int64intfloatdyn2d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, triangle, float, 3D
  m.def("intersects_mesh_edge_mesh_int64intfloat33d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, dynamic, float, 3D
  m.def("intersects_mesh_edge_mesh_int64intfloatdyn3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, triangle, double, 2D
  m.def("intersects_mesh_edge_mesh_int64intdouble32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, dynamic, double, 2D
  m.def("intersects_mesh_edge_mesh_int64intdoubledyn2d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, triangle, double, 3D
  m.def("intersects_mesh_edge_mesh_int64intdouble33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, dynamic, double, 3D
  m.def("intersects_mesh_edge_mesh_int64intdoubledyn3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));
}

} // namespace tf::py

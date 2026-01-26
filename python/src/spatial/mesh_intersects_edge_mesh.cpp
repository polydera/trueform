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
  m.def("intersects_mesh_edge_mesh_intint3float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, triangle, float, 2D
  m.def("intersects_mesh_edge_mesh_intint643float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, triangle, float, 2D
  m.def("intersects_mesh_edge_mesh_int64int643float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int32 edge_mesh, dynamic, float, 2D
  m.def("intersects_mesh_edge_mesh_intintdynfloat2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, dynamic, float, 2D
  m.def("intersects_mesh_edge_mesh_intint64dynfloat2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, dynamic, float, 2D
  m.def("intersects_mesh_edge_mesh_int64int64dynfloat2d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // ==== float, 3D ====

  // int32 mesh, int32 edge_mesh, triangle, float, 3D
  m.def("intersects_mesh_edge_mesh_intint3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, triangle, float, 3D
  m.def("intersects_mesh_edge_mesh_intint643float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, triangle, float, 3D
  m.def("intersects_mesh_edge_mesh_int64int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int32 edge_mesh, dynamic, float, 3D
  m.def("intersects_mesh_edge_mesh_intintdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, dynamic, float, 3D
  m.def("intersects_mesh_edge_mesh_intint64dynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, dynamic, float, 3D
  m.def("intersects_mesh_edge_mesh_int64int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // ==== double, 2D ====

  // int32 mesh, int32 edge_mesh, triangle, double, 2D
  m.def("intersects_mesh_edge_mesh_intint3double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, triangle, double, 2D
  m.def("intersects_mesh_edge_mesh_intint643double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, triangle, double, 2D
  m.def("intersects_mesh_edge_mesh_int64int643double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int32 edge_mesh, dynamic, double, 2D
  m.def("intersects_mesh_edge_mesh_intintdyndouble2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, dynamic, double, 2D
  m.def("intersects_mesh_edge_mesh_intint64dyndouble2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, dynamic, double, 2D
  m.def("intersects_mesh_edge_mesh_int64int64dyndouble2d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // ==== double, 3D ====

  // int32 mesh, int32 edge_mesh, triangle, double, 3D
  m.def("intersects_mesh_edge_mesh_intint3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, triangle, double, 3D
  m.def("intersects_mesh_edge_mesh_intint643double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, triangle, double, 3D
  m.def("intersects_mesh_edge_mesh_int64int643double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int32 edge_mesh, dynamic, double, 3D
  m.def("intersects_mesh_edge_mesh_intintdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int32 mesh, int64 edge_mesh, dynamic, double, 3D
  m.def("intersects_mesh_edge_mesh_intint64dyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int64 edge_mesh, dynamic, double, 3D
  m.def("intersects_mesh_edge_mesh_int64int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, triangle, float, 2D
  m.def("intersects_mesh_edge_mesh_int64int3float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, dynamic, float, 2D
  m.def("intersects_mesh_edge_mesh_int64intdynfloat2d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, triangle, float, 3D
  m.def("intersects_mesh_edge_mesh_int64int3float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, dynamic, float, 3D
  m.def("intersects_mesh_edge_mesh_int64intdynfloat3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, triangle, double, 2D
  m.def("intersects_mesh_edge_mesh_int64int3double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, dynamic, double, 2D
  m.def("intersects_mesh_edge_mesh_int64intdyndouble2d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, triangle, double, 3D
  m.def("intersects_mesh_edge_mesh_int64int3double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));

  // int64 mesh, int32 edge_mesh, dynamic, double, 3D
  m.def("intersects_mesh_edge_mesh_int64intdyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh) {
          return form_intersects_form(mesh, edge_mesh);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"));
}

} // namespace tf::py

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
#include <trueform/python/spatial/form_intersects_form.hpp>

namespace tf::py {

auto register_mesh_intersects_mesh(nanobind::module_ &m) -> void {

  // ============================================================================
  // Mesh intersects Mesh
  // Index types: int32, int64 (can differ)
  // Ngons: 3 (triangle), dynamic (can differ)
  // Real types: float, double (must match)
  // Dims: 2D, 3D (must match)
  // Total: 2 × 2 × 2 × 2 × 2 × 2 = 64 functions
  // Format: mesh0_index mesh1_index mesh0_ngon mesh1_ngon
  // ============================================================================

  // ==== float, 2D ====
  
  // int32 × int32
  m.def("intersects_mesh_mesh_intint33float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int, float, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint3dynfloat2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int, float, dynamic_size, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intintdyn3float2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh0,
           mesh_wrapper<int, float, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intintdyndynfloat2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh0,
           mesh_wrapper<int, float, dynamic_size, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int32 × int64
  m.def("intersects_mesh_mesh_intint6433float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint643dynfloat2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint64dyn3float2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint64dyndynfloat2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int64 × int64
  m.def("intersects_mesh_mesh_int64int6433float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int643dynfloat2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int64dyn3float2d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int64dyndynfloat2d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // ==== float, 3D ====
  
  // int32 × int32
  m.def("intersects_mesh_mesh_intint33float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint3dynfloat3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, dynamic_size, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intintdyn3float3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intintdyndynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh0,
           mesh_wrapper<int, float, dynamic_size, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int32 × int64
  m.def("intersects_mesh_mesh_intint6433float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint643dynfloat3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint64dyn3float3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint64dyndynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int64 × int64
  m.def("intersects_mesh_mesh_int64int6433float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int643dynfloat3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int64dyn3float3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int64dyndynfloat3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // ==== double, 2D ====
  
  // int32 × int32
  m.def("intersects_mesh_mesh_intint33double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int, double, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint3dyndouble2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int, double, dynamic_size, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intintdyn3double2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh0,
           mesh_wrapper<int, double, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intintdyndyndouble2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh0,
           mesh_wrapper<int, double, dynamic_size, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int32 × int64
  m.def("intersects_mesh_mesh_intint6433double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint643dyndouble2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint64dyn3double2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint64dyndyndouble2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int64 × int64
  m.def("intersects_mesh_mesh_int64int6433double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int643dyndouble2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int64dyn3double2d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int64dyndyndouble2d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // ==== double, 3D ====
  
  // int32 × int32
  m.def("intersects_mesh_mesh_intint33double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint3dyndouble3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, dynamic_size, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intintdyn3double3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intintdyndyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int, double, dynamic_size, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int32 × int64
  m.def("intersects_mesh_mesh_intint6433double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint643dyndouble3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint64dyn3double3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint64dyndyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int64 × int64
  m.def("intersects_mesh_mesh_int64int6433double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int643dyndouble3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int64dyn3double3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int64dyndyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });
}

} // namespace tf::py

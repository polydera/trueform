/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/form_intersects_form.hpp>

namespace tf::py {

auto register_mesh_intersects_mesh(nanobind::module_ &m) -> void {

  // ============================================================================
  // Mesh intersects Mesh
  // Index types: int32, int64 (can differ)
  // Ngons: 3 (triangle), 4 (quad) (can differ)
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

  m.def("intersects_mesh_mesh_intint34float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int, float, 4, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint43float2d",
        [](mesh_wrapper<int, float, 4, 2> &mesh0,
           mesh_wrapper<int, float, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint44float2d",
        [](mesh_wrapper<int, float, 4, 2> &mesh0,
           mesh_wrapper<int, float, 4, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int32 × int64
  m.def("intersects_mesh_mesh_intint6433float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint6434float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, 4, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint6443float2d",
        [](mesh_wrapper<int, float, 4, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint6444float2d",
        [](mesh_wrapper<int, float, 4, 2> &mesh0,
           mesh_wrapper<int64_t, float, 4, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int64 × int64
  m.def("intersects_mesh_mesh_int64int6433float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int6434float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, 4, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int6443float2d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int6444float2d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh0,
           mesh_wrapper<int64_t, float, 4, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // ==== float, 3D ====
  
  // int32 × int32
  m.def("intersects_mesh_mesh_intint33float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint34float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, 4, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint43float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint44float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int, float, 4, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int32 × int64
  m.def("intersects_mesh_mesh_intint6433float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint6434float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint6443float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint6444float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int64 × int64
  m.def("intersects_mesh_mesh_int64int6433float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int6434float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int6443float3d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int6444float3d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // ==== double, 2D ====
  
  // int32 × int32
  m.def("intersects_mesh_mesh_intint33double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int, double, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint34double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int, double, 4, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint43double2d",
        [](mesh_wrapper<int, double, 4, 2> &mesh0,
           mesh_wrapper<int, double, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint44double2d",
        [](mesh_wrapper<int, double, 4, 2> &mesh0,
           mesh_wrapper<int, double, 4, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int32 × int64
  m.def("intersects_mesh_mesh_intint6433double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint6434double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 4, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint6443double2d",
        [](mesh_wrapper<int, double, 4, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint6444double2d",
        [](mesh_wrapper<int, double, 4, 2> &mesh0,
           mesh_wrapper<int64_t, double, 4, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int64 × int64
  m.def("intersects_mesh_mesh_int64int6433double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int6434double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 4, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int6443double2d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int6444double2d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh0,
           mesh_wrapper<int64_t, double, 4, 2> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // ==== double, 3D ====
  
  // int32 × int32
  m.def("intersects_mesh_mesh_intint33double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint34double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 4, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint43double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint44double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int, double, 4, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int32 × int64
  m.def("intersects_mesh_mesh_intint6433double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint6434double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint6443double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_intint6444double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  // int64 × int64
  m.def("intersects_mesh_mesh_int64int6433double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int6434double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int6443double3d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });

  m.def("intersects_mesh_mesh_int64int6444double3d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1) {
          return form_intersects_form(mesh0, mesh1);
        });
}

} // namespace tf::py

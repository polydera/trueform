/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include "trueform/python/intersect/intersection_curves.hpp"

namespace tf::py {

auto register_intersect_intersection_curves(nanobind::module_ &m) -> void {

  // ============================================================================
  // Mesh intersection curves with Mesh
  // Index types: int32, int64 (can differ)
  // Ngons: 3 (triangle), 4 (quad) (can differ)
  // Real types: float, double (must match)
  // Dims: 3D only (booleans only supported in 3D)
  // Total: 3 × 4 × 2 = 24 functions
  // Format: mesh0_index mesh1_index mesh0_ngon mesh1_ngon
  // Note: int64 × int32 is handled by Python layer symmetry
  // ============================================================================

  // ==== float, 3D ====

  // int32 × int32
  m.def("intersection_curves_mesh_mesh_intint33float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint34float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, 4, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint43float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint44float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int, float, 4, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  // int32 × int64
  m.def("intersection_curves_mesh_mesh_intint6433float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint6434float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint6443float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint6444float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  // int64 × int64
  m.def("intersection_curves_mesh_mesh_int64int6433float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_int64int6434float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_int64int6443float3d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_int64int6444float3d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  // ==== double, 3D ====

  // int32 × int32
  m.def("intersection_curves_mesh_mesh_intint33double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint34double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 4, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint43double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint44double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int, double, 4, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  // int32 × int64
  m.def("intersection_curves_mesh_mesh_intint6433double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint6434double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint6443double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_intint6444double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  // int64 × int64
  m.def("intersection_curves_mesh_mesh_int64int6433double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_int64int6434double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_int64int6443double3d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });

  m.def("intersection_curves_mesh_mesh_int64int6444double3d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1) {
          return intersection_curves(mesh0, mesh1);
        });
}

} // namespace tf::py

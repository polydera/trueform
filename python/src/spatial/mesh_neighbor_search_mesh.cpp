/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/form_form_neighbor_search.hpp>

namespace tf::py {

auto register_mesh_neighbor_search_mesh(nanobind::module_ &m) -> void {

  // ============================================================================
  // Mesh neighbor_search Mesh
  // Index types: int32, int64 (can differ)
  // Ngons: 3 (triangle), 4 (quad) (can differ)
  // Real types: float, double (must match)
  // Dims: 2D, 3D (must match)
  // Canonical ordering: int×int, int×int64, int64×int64 (3 combinations only)
  // Total: 3 × 4 × 2 × 2 = 48 functions
  // Format: mesh0_index mesh1_index mesh0_ngon mesh1_ngon
  // ============================================================================

  // ==== float, 2D ====

  // int32 × int32
  m.def("neighbor_search_mesh_mesh_intint33float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int, float, 3, 2> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint34float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int, float, 4, 2> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint43float2d",
        [](mesh_wrapper<int, float, 4, 2> &mesh0,
           mesh_wrapper<int, float, 3, 2> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint44float2d",
        [](mesh_wrapper<int, float, 4, 2> &mesh0,
           mesh_wrapper<int, float, 4, 2> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 × int64
  m.def("neighbor_search_mesh_mesh_intint6433float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint6434float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, 4, 2> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint6443float2d",
        [](mesh_wrapper<int, float, 4, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint6444float2d",
        [](mesh_wrapper<int, float, 4, 2> &mesh0,
           mesh_wrapper<int64_t, float, 4, 2> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 × int64
  m.def("neighbor_search_mesh_mesh_int64int6433float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int6434float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, 4, 2> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int6443float2d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int6444float2d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh0,
           mesh_wrapper<int64_t, float, 4, 2> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // ==== float, 3D ====

  // int32 × int32
  m.def("neighbor_search_mesh_mesh_intint33float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint34float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, 4, 3> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint43float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint44float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int, float, 4, 3> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 × int64
  m.def("neighbor_search_mesh_mesh_intint6433float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint6434float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint6443float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint6444float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 × int64
  m.def("neighbor_search_mesh_mesh_int64int6433float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int6434float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int6443float3d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int6444float3d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // ==== double, 2D ====

  // int32 × int32
  m.def("neighbor_search_mesh_mesh_intint33double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int, double, 3, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint34double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int, double, 4, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint43double2d",
        [](mesh_wrapper<int, double, 4, 2> &mesh0,
           mesh_wrapper<int, double, 3, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint44double2d",
        [](mesh_wrapper<int, double, 4, 2> &mesh0,
           mesh_wrapper<int, double, 4, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 × int64
  m.def("neighbor_search_mesh_mesh_intint6433double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint6434double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 4, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint6443double2d",
        [](mesh_wrapper<int, double, 4, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint6444double2d",
        [](mesh_wrapper<int, double, 4, 2> &mesh0,
           mesh_wrapper<int64_t, double, 4, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 × int64
  m.def("neighbor_search_mesh_mesh_int64int6433double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int6434double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 4, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int6443double2d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int6444double2d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh0,
           mesh_wrapper<int64_t, double, 4, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // ==== double, 3D ====

  // int32 × int32
  m.def("neighbor_search_mesh_mesh_intint33double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint34double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 4, 3> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint43double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint44double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int, double, 4, 3> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 × int64
  m.def("neighbor_search_mesh_mesh_intint6433double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint6434double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint6443double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint6444double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 × int64
  m.def("neighbor_search_mesh_mesh_int64int6433double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int6434double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int6443double3d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int6444double3d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // Note: int64 × int32 is handled by Python canonical ordering
}

} // namespace tf::py

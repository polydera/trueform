/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <trueform/python/core/mesh.hpp>
#include <trueform/python/spatial/form_form_gather_ids.hpp>

namespace tf::py {

auto register_mesh_gather_ids_mesh(nanobind::module_ &m) -> void {

  // ============================================================================
  // Mesh gather_ids Mesh
  // Index types: int32, int64 (can differ)
  // Real types: float, double (must match)
  // Ngon: 3, 4 (for both meshes, can differ)
  // Dims: 2D, 3D (must match)
  // Total: 64 functions (48 canonical + 16 for int64×int32)
  // ============================================================================

  // int × int, 3×3, float, 2D
  m.def("gather_ids_mesh_mesh_intint33float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int, float, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 3×3, float, 2D
  m.def("gather_ids_mesh_mesh_intint6433float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 3×3, float, 2D
  m.def("gather_ids_mesh_mesh_int64int6433float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 3×4, float, 2D
  m.def("gather_ids_mesh_mesh_intint34float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int, float, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 3×4, float, 2D
  m.def("gather_ids_mesh_mesh_intint6434float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 3×4, float, 2D
  m.def("gather_ids_mesh_mesh_int64int6434float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh0,
           mesh_wrapper<int64_t, float, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 4×3, float, 2D
  m.def("gather_ids_mesh_mesh_intint43float2d",
        [](mesh_wrapper<int, float, 4, 2> &mesh0,
           mesh_wrapper<int, float, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 4×3, float, 2D
  m.def("gather_ids_mesh_mesh_intint6443float2d",
        [](mesh_wrapper<int, float, 4, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 4×3, float, 2D
  m.def("gather_ids_mesh_mesh_int64int6443float2d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh0,
           mesh_wrapper<int64_t, float, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 4×4, float, 2D
  m.def("gather_ids_mesh_mesh_intint44float2d",
        [](mesh_wrapper<int, float, 4, 2> &mesh0,
           mesh_wrapper<int, float, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 4×4, float, 2D
  m.def("gather_ids_mesh_mesh_intint6444float2d",
        [](mesh_wrapper<int, float, 4, 2> &mesh0,
           mesh_wrapper<int64_t, float, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 4×4, float, 2D
  m.def("gather_ids_mesh_mesh_int64int6444float2d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh0,
           mesh_wrapper<int64_t, float, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 3×3, float, 3D
  m.def("gather_ids_mesh_mesh_intint33float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 3×3, float, 3D
  m.def("gather_ids_mesh_mesh_intint6433float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 3×3, float, 3D
  m.def("gather_ids_mesh_mesh_int64int6433float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 3×4, float, 3D
  m.def("gather_ids_mesh_mesh_intint34float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 3×4, float, 3D
  m.def("gather_ids_mesh_mesh_intint6434float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 3×4, float, 3D
  m.def("gather_ids_mesh_mesh_int64int6434float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 4×3, float, 3D
  m.def("gather_ids_mesh_mesh_intint43float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 4×3, float, 3D
  m.def("gather_ids_mesh_mesh_intint6443float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 4×3, float, 3D
  m.def("gather_ids_mesh_mesh_int64int6443float3d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 4×4, float, 3D
  m.def("gather_ids_mesh_mesh_intint44float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int, float, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 4×4, float, 3D
  m.def("gather_ids_mesh_mesh_intint6444float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 4×4, float, 3D
  m.def("gather_ids_mesh_mesh_int64int6444float3d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh0,
           mesh_wrapper<int64_t, float, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 3×3, double, 2D
  m.def("gather_ids_mesh_mesh_intint33double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int, double, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 3×3, double, 2D
  m.def("gather_ids_mesh_mesh_intint6433double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 3×3, double, 2D
  m.def("gather_ids_mesh_mesh_int64int6433double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 3×4, double, 2D
  m.def("gather_ids_mesh_mesh_intint34double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int, double, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 3×4, double, 2D
  m.def("gather_ids_mesh_mesh_intint6434double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 3×4, double, 2D
  m.def("gather_ids_mesh_mesh_int64int6434double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 4×3, double, 2D
  m.def("gather_ids_mesh_mesh_intint43double2d",
        [](mesh_wrapper<int, double, 4, 2> &mesh0,
           mesh_wrapper<int, double, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 4×3, double, 2D
  m.def("gather_ids_mesh_mesh_intint6443double2d",
        [](mesh_wrapper<int, double, 4, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 4×3, double, 2D
  m.def("gather_ids_mesh_mesh_int64int6443double2d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 4×4, double, 2D
  m.def("gather_ids_mesh_mesh_intint44double2d",
        [](mesh_wrapper<int, double, 4, 2> &mesh0,
           mesh_wrapper<int, double, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 4×4, double, 2D
  m.def("gather_ids_mesh_mesh_intint6444double2d",
        [](mesh_wrapper<int, double, 4, 2> &mesh0,
           mesh_wrapper<int64_t, double, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 4×4, double, 2D
  m.def("gather_ids_mesh_mesh_int64int6444double2d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh0,
           mesh_wrapper<int64_t, double, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 3×3, double, 3D
  m.def("gather_ids_mesh_mesh_intint33double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 3×3, double, 3D
  m.def("gather_ids_mesh_mesh_intint6433double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 3×3, double, 3D
  m.def("gather_ids_mesh_mesh_int64int6433double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 3×4, double, 3D
  m.def("gather_ids_mesh_mesh_intint34double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 3×4, double, 3D
  m.def("gather_ids_mesh_mesh_intint6434double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 3×4, double, 3D
  m.def("gather_ids_mesh_mesh_int64int6434double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 4×3, double, 3D
  m.def("gather_ids_mesh_mesh_intint43double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 4×3, double, 3D
  m.def("gather_ids_mesh_mesh_intint6443double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 4×3, double, 3D
  m.def("gather_ids_mesh_mesh_int64int6443double3d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, 4×4, double, 3D
  m.def("gather_ids_mesh_mesh_intint44double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int, double, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 4×4, double, 3D
  m.def("gather_ids_mesh_mesh_intint6444double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 4×4, double, 3D
  m.def("gather_ids_mesh_mesh_int64int6444double3d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh0,
           mesh_wrapper<int64_t, double, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 3×3, float, 2D
  m.def("gather_ids_mesh_mesh_int64int33float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh0,
           mesh_wrapper<int, float, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 3×4, float, 2D
  m.def("gather_ids_mesh_mesh_int64int34float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh0,
           mesh_wrapper<int, float, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 4×3, float, 2D
  m.def("gather_ids_mesh_mesh_int64int43float2d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh0,
           mesh_wrapper<int, float, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 4×4, float, 2D
  m.def("gather_ids_mesh_mesh_int64int44float2d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh0,
           mesh_wrapper<int, float, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 3×3, float, 3D
  m.def("gather_ids_mesh_mesh_int64int33float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 3×4, float, 3D
  m.def("gather_ids_mesh_mesh_int64int34float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 4×3, float, 3D
  m.def("gather_ids_mesh_mesh_int64int43float3d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 4×4, float, 3D
  m.def("gather_ids_mesh_mesh_int64int44float3d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh0,
           mesh_wrapper<int, float, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 3×3, double, 2D
  m.def("gather_ids_mesh_mesh_int64int33double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh0,
           mesh_wrapper<int, double, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 3×4, double, 2D
  m.def("gather_ids_mesh_mesh_int64int34double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh0,
           mesh_wrapper<int, double, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 4×3, double, 2D
  m.def("gather_ids_mesh_mesh_int64int43double2d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh0,
           mesh_wrapper<int, double, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 4×4, double, 2D
  m.def("gather_ids_mesh_mesh_int64int44double2d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh0,
           mesh_wrapper<int, double, 4, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 3×3, double, 3D
  m.def("gather_ids_mesh_mesh_int64int33double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 3×4, double, 3D
  m.def("gather_ids_mesh_mesh_int64int34double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh0,
           mesh_wrapper<int, double, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 4×3, double, 3D
  m.def("gather_ids_mesh_mesh_int64int43double3d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh0,
           mesh_wrapper<int, double, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, 4×4, double, 3D
  m.def("gather_ids_mesh_mesh_int64int44double3d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh0,
           mesh_wrapper<int, double, 4, 3> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // Total: 64 functions
}

} // namespace tf::py

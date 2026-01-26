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
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <trueform/python/spatial/edge_mesh.hpp>
#include <trueform/python/spatial/form_form_gather_ids.hpp>

namespace tf::py {

auto register_edge_mesh_gather_ids_edge_mesh(nanobind::module_ &m) -> void {

  // ============================================================================
  // EdgeMesh gather_ids EdgeMesh
  // Index types: int32, int64 (can differ)
  // Real types: float, double (must match)
  // Dims: 2D, 3D (must match)
  // Total: 3 × 2 × 2 = 12 functions
  // ============================================================================

  // int32 × int32, float, 2D
  m.def("gather_ids_edge_mesh_edge_mesh_intintfloat2d",
        [](edge_mesh_wrapper<int, float, 2> &edge_mesh0,
           edge_mesh_wrapper<int, float, 2> &edge_mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(edge_mesh0, edge_mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("edge_mesh0"), nanobind::arg("edge_mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int32 × int64, float, 2D
  m.def("gather_ids_edge_mesh_edge_mesh_intint64float2d",
        [](edge_mesh_wrapper<int, float, 2> &edge_mesh0,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(edge_mesh0, edge_mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("edge_mesh0"), nanobind::arg("edge_mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, float, 2D
  m.def("gather_ids_edge_mesh_edge_mesh_int64int64float2d",
        [](edge_mesh_wrapper<int64_t, float, 2> &edge_mesh0,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(edge_mesh0, edge_mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("edge_mesh0"), nanobind::arg("edge_mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int32 × int32, float, 3D
  m.def("gather_ids_edge_mesh_edge_mesh_intintfloat3d",
        [](edge_mesh_wrapper<int, float, 3> &edge_mesh0,
           edge_mesh_wrapper<int, float, 3> &edge_mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(edge_mesh0, edge_mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("edge_mesh0"), nanobind::arg("edge_mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int32 × int64, float, 3D
  m.def("gather_ids_edge_mesh_edge_mesh_intint64float3d",
        [](edge_mesh_wrapper<int, float, 3> &edge_mesh0,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(edge_mesh0, edge_mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("edge_mesh0"), nanobind::arg("edge_mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, float, 3D
  m.def("gather_ids_edge_mesh_edge_mesh_int64int64float3d",
        [](edge_mesh_wrapper<int64_t, float, 3> &edge_mesh0,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(edge_mesh0, edge_mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("edge_mesh0"), nanobind::arg("edge_mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int32 × int32, double, 2D
  m.def("gather_ids_edge_mesh_edge_mesh_intintdouble2d",
        [](edge_mesh_wrapper<int, double, 2> &edge_mesh0,
           edge_mesh_wrapper<int, double, 2> &edge_mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(edge_mesh0, edge_mesh1,
                                                  predicate_type, threshold);
        },
        nanobind::arg("edge_mesh0"), nanobind::arg("edge_mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int32 × int64, double, 2D
  m.def("gather_ids_edge_mesh_edge_mesh_intint64double2d",
        [](edge_mesh_wrapper<int, double, 2> &edge_mesh0,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(edge_mesh0, edge_mesh1,
                                                  predicate_type, threshold);
        },
        nanobind::arg("edge_mesh0"), nanobind::arg("edge_mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, double, 2D
  m.def("gather_ids_edge_mesh_edge_mesh_int64int64double2d",
        [](edge_mesh_wrapper<int64_t, double, 2> &edge_mesh0,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(edge_mesh0, edge_mesh1,
                                                  predicate_type, threshold);
        },
        nanobind::arg("edge_mesh0"), nanobind::arg("edge_mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int32 × int32, double, 3D
  m.def("gather_ids_edge_mesh_edge_mesh_intintdouble3d",
        [](edge_mesh_wrapper<int, double, 3> &edge_mesh0,
           edge_mesh_wrapper<int, double, 3> &edge_mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(edge_mesh0, edge_mesh1,
                                                  predicate_type, threshold);
        },
        nanobind::arg("edge_mesh0"), nanobind::arg("edge_mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int32 × int64, double, 3D
  m.def("gather_ids_edge_mesh_edge_mesh_intint64double3d",
        [](edge_mesh_wrapper<int, double, 3> &edge_mesh0,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(edge_mesh0, edge_mesh1,
                                                  predicate_type, threshold);
        },
        nanobind::arg("edge_mesh0"), nanobind::arg("edge_mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, double, 3D
  m.def("gather_ids_edge_mesh_edge_mesh_int64int64double3d",
        [](edge_mesh_wrapper<int64_t, double, 3> &edge_mesh0,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(edge_mesh0, edge_mesh1,
                                                  predicate_type, threshold);
        },
        nanobind::arg("edge_mesh0"), nanobind::arg("edge_mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // Note: int64 × int32 is handled by dispatch table symmetry
}

} // namespace tf::py

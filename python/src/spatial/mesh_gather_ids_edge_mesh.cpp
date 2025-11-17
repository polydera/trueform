/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <trueform/python/spatial/edge_mesh.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/form_form_gather_ids.hpp>

namespace tf::py {

auto register_mesh_gather_ids_edge_mesh(nanobind::module_ &m) -> void {

  // ============================================================================
  // Mesh gather_ids EdgeMesh
  // Index types: int32, int64 (can differ)
  // Real types: float, double (must match)
  // Ngon: 3, 4 (for mesh)
  // Dims: 2D, 3D (must match)
  // Total: 32 functions (24 canonical + 8 for int64×int32)
  // ============================================================================

  // int × int, ngon=3, float, 2D
  m.def("gather_ids_mesh_edge_mesh_intint3float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, ngon=3, float, 2D
  m.def("gather_ids_mesh_edge_mesh_intint643float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, ngon=3, float, 2D
  m.def("gather_ids_mesh_edge_mesh_int64int643float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, ngon=4, float, 2D
  m.def("gather_ids_mesh_edge_mesh_intint4float2d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, ngon=4, float, 2D
  m.def("gather_ids_mesh_edge_mesh_intint644float2d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, ngon=4, float, 2D
  m.def("gather_ids_mesh_edge_mesh_int64int644float2d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, ngon=3, float, 3D
  m.def("gather_ids_mesh_edge_mesh_intint3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, ngon=3, float, 3D
  m.def("gather_ids_mesh_edge_mesh_intint643float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, ngon=3, float, 3D
  m.def("gather_ids_mesh_edge_mesh_int64int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, ngon=4, float, 3D
  m.def("gather_ids_mesh_edge_mesh_intint4float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, ngon=4, float, 3D
  m.def("gather_ids_mesh_edge_mesh_intint644float3d",
        [](mesh_wrapper<int, float, 4, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, ngon=4, float, 3D
  m.def("gather_ids_mesh_edge_mesh_int64int644float3d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, ngon=3, double, 2D
  m.def("gather_ids_mesh_edge_mesh_intint3double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, ngon=3, double, 2D
  m.def("gather_ids_mesh_edge_mesh_intint643double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, ngon=3, double, 2D
  m.def("gather_ids_mesh_edge_mesh_int64int643double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, ngon=4, double, 2D
  m.def("gather_ids_mesh_edge_mesh_intint4double2d",
        [](mesh_wrapper<int, double, 4, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, ngon=4, double, 2D
  m.def("gather_ids_mesh_edge_mesh_intint644double2d",
        [](mesh_wrapper<int, double, 4, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, ngon=4, double, 2D
  m.def("gather_ids_mesh_edge_mesh_int64int644double2d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, ngon=3, double, 3D
  m.def("gather_ids_mesh_edge_mesh_intint3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, ngon=3, double, 3D
  m.def("gather_ids_mesh_edge_mesh_intint643double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, ngon=3, double, 3D
  m.def("gather_ids_mesh_edge_mesh_int64int643double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, ngon=4, double, 3D
  m.def("gather_ids_mesh_edge_mesh_intint4double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, ngon=4, double, 3D
  m.def("gather_ids_mesh_edge_mesh_intint644double3d",
        [](mesh_wrapper<int, double, 4, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, ngon=4, double, 3D
  m.def("gather_ids_mesh_edge_mesh_int64int644double3d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, ngon=3, float, 2D
  m.def("gather_ids_mesh_edge_mesh_int64int3float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, ngon=4, float, 2D
  m.def("gather_ids_mesh_edge_mesh_int64int4float2d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, ngon=3, float, 3D
  m.def("gather_ids_mesh_edge_mesh_int64int3float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, ngon=4, float, 3D
  m.def("gather_ids_mesh_edge_mesh_int64int4float3d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, ngon=3, double, 2D
  m.def("gather_ids_mesh_edge_mesh_int64int3double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, ngon=4, double, 2D
  m.def("gather_ids_mesh_edge_mesh_int64int4double2d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, ngon=3, double, 3D
  m.def("gather_ids_mesh_edge_mesh_int64int3double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, ngon=4, double, 3D
  m.def("gather_ids_mesh_edge_mesh_int64int4double3d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // Total: 32 functions
}

} // namespace tf::py

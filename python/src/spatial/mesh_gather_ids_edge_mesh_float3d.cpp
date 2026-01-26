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
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/form_form_gather_ids.hpp>

namespace tf::py {

auto register_mesh_gather_ids_edge_mesh_float3d(nanobind::module_ &m) -> void {
  // float, 3D variants (8 functions)

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

  // int × int, dynamic, float, 3D
  m.def("gather_ids_mesh_edge_mesh_intintdynfloat3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, dynamic, float, 3D
  m.def("gather_ids_mesh_edge_mesh_intint64dynfloat3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, dynamic, float, 3D
  m.def("gather_ids_mesh_edge_mesh_int64int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh, edge_mesh,
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

  // int64 × int, dynamic, float, 3D
  m.def("gather_ids_mesh_edge_mesh_int64intdynfloat3d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());
}

} // namespace tf::py

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
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/form_form_gather_ids.hpp>

namespace tf::py {

auto register_mesh_gather_ids_mesh_float3d(nanobind::module_ &m) -> void {
  // float, 3D variants (16 functions)

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

  // int × int, 3×dyn, float, 3D
  m.def("gather_ids_mesh_mesh_intint3dynfloat3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 3×dyn, float, 3D
  m.def("gather_ids_mesh_mesh_intint643dynfloat3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 3×dyn, float, 3D
  m.def("gather_ids_mesh_mesh_int64int643dynfloat3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, dyn×3, float, 3D
  m.def("gather_ids_mesh_mesh_intintdyn3float3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, dyn×3, float, 3D
  m.def("gather_ids_mesh_mesh_intint64dyn3float3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, dyn×3, float, 3D
  m.def("gather_ids_mesh_mesh_int64int64dyn3float3d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, float, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, dyn×dyn, float, 3D
  m.def("gather_ids_mesh_mesh_intintdyndynfloat3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh0,
           mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, dyn×dyn, float, 3D
  m.def("gather_ids_mesh_mesh_intint64dyndynfloat3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, dyn×dyn, float, 3D
  m.def("gather_ids_mesh_mesh_int64int64dyndynfloat3d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh0,
           mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
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

  // int64 × int, 3×dyn, float, 3D
  m.def("gather_ids_mesh_mesh_int64int3dynfloat3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh0,
           mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, dyn×3, float, 3D
  m.def("gather_ids_mesh_mesh_int64intdyn3float3d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh0,
           mesh_wrapper<int, float, 3, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, dyn×dyn, float, 3D
  m.def("gather_ids_mesh_mesh_int64intdyndynfloat3d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh0,
           mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh1,
           const std::string &predicate_type, std::optional<float> threshold) {
          return form_form_gather_ids<float, 3>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());
}

} // namespace tf::py

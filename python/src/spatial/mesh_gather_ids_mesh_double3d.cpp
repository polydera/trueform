/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/form_form_gather_ids.hpp>

namespace tf::py {

auto register_mesh_gather_ids_mesh_double3d(nanobind::module_ &m) -> void {
  // double, 3D variants (16 functions)

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
}

} // namespace tf::py

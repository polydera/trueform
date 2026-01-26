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

auto register_mesh_gather_ids_mesh_double2d(nanobind::module_ &m) -> void {
  // double, 2D variants (16 functions)

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

  // int × int, 3×dyn, double, 2D
  m.def("gather_ids_mesh_mesh_intint3dyndouble2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, 3×dyn, double, 2D
  m.def("gather_ids_mesh_mesh_intint643dyndouble2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, 3×dyn, double, 2D
  m.def("gather_ids_mesh_mesh_int64int643dyndouble2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, dyn×3, double, 2D
  m.def("gather_ids_mesh_mesh_intintdyn3double2d",
        [](mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh0,
           mesh_wrapper<int, double, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, dyn×3, double, 2D
  m.def("gather_ids_mesh_mesh_intint64dyn3double2d",
        [](mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, dyn×3, double, 2D
  m.def("gather_ids_mesh_mesh_int64int64dyn3double2d",
        [](mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int, dyn×dyn, double, 2D
  m.def("gather_ids_mesh_mesh_intintdyndyndouble2d",
        [](mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh0,
           mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, dyn×dyn, double, 2D
  m.def("gather_ids_mesh_mesh_intint64dyndyndouble2d",
        [](mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, dyn×dyn, double, 2D
  m.def("gather_ids_mesh_mesh_int64int64dyndyndouble2d",
        [](mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
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

  // int64 × int, 3×dyn, double, 2D
  m.def("gather_ids_mesh_mesh_int64int3dyndouble2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh0,
           mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, dyn×3, double, 2D
  m.def("gather_ids_mesh_mesh_int64intdyn3double2d",
        [](mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh0,
           mesh_wrapper<int, double, 3, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int, dyn×dyn, double, 2D
  m.def("gather_ids_mesh_mesh_int64intdyndyndouble2d",
        [](mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh0,
           mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh1,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 2>(mesh0, mesh1,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh0"), nanobind::arg("mesh1"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());
}

} // namespace tf::py

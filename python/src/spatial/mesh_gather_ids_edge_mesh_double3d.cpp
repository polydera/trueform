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

auto register_mesh_gather_ids_edge_mesh_double3d(nanobind::module_ &m) -> void {
  // double, 3D variants (8 functions)

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

  // int × int, dynamic, double, 3D
  m.def("gather_ids_mesh_edge_mesh_intintdyndouble3d",
        [](mesh_wrapper<int, double, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int × int64, dynamic, double, 3D
  m.def("gather_ids_mesh_edge_mesh_intint64dyndouble3d",
        [](mesh_wrapper<int, double, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());

  // int64 × int64, dynamic, double, 3D
  m.def("gather_ids_mesh_edge_mesh_int64int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh, edge_mesh,
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

  // int64 × int, dynamic, double, 3D
  m.def("gather_ids_mesh_edge_mesh_int64intdyndouble3d",
        [](mesh_wrapper<int64_t, double, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh,
           const std::string &predicate_type, std::optional<double> threshold) {
          return form_form_gather_ids<double, 3>(mesh, edge_mesh,
                                                 predicate_type, threshold);
        },
        nanobind::arg("mesh"), nanobind::arg("edge_mesh"),
        nanobind::arg("predicate_type"),
        nanobind::arg("threshold").none() = nanobind::none());
}

} // namespace tf::py

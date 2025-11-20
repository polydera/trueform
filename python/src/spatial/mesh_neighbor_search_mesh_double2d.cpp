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

auto register_mesh_neighbor_search_mesh_double2d(nanobind::module_ &m) -> void {

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
}

} // namespace tf::py

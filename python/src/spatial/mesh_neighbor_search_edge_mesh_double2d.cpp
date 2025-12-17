/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <trueform/python/spatial/edge_mesh.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/form_form_neighbor_search.hpp>

namespace tf::py {

auto register_mesh_neighbor_search_edge_mesh_double2d(nanobind::module_ &m) -> void {

  // ==== double, 2D ====

  // int32 mesh, int32 edge_mesh, triangle, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_intintdouble32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, triangle, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_intint64double32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, triangle, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64int64double32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int32 edge_mesh, dynamic, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_intintdoubledyn2d",
        [](mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, dynamic, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_intint64doubledyn2d",
        [](mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, dynamic, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64int64doubledyn2d",
        [](mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, triangle, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64intdouble32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, dynamic, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64intdoubledyn2d",
        [](mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());
}

} // namespace tf::py

/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (www.trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <trueform/python/spatial/edge_mesh.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/form_form_neighbor_search.hpp>

namespace tf::py {

auto register_mesh_neighbor_search_edge_mesh_double3d(nanobind::module_ &m) -> void {

  // ==== double, 3D ====

  // int32 mesh, int32 edge_mesh, triangle, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_intintdouble33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, triangle, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_intint64double33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, triangle, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64int64double33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int32 edge_mesh, dynamic, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_intintdoubledyn3d",
        [](mesh_wrapper<int, double, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, dynamic, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_intint64doubledyn3d",
        [](mesh_wrapper<int, double, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, dynamic, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64int64doubledyn3d",
        [](mesh_wrapper<int64_t, double, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, triangle, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64intdouble33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, dynamic, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64intdoubledyn3d",
        [](mesh_wrapper<int64_t, double, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());
}

} // namespace tf::py

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

auto register_mesh_neighbor_search_edge_mesh_float2d(nanobind::module_ &m) -> void {

  // ==== float, 2D ====

  // int32 mesh, int32 edge_mesh, triangle, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_intintfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, triangle, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_intint64float32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, triangle, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int32 edge_mesh, dynamic, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_intintfloatdyn2d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, dynamic, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_intint64floatdyn2d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, dynamic, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64int64floatdyn2d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, triangle, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64intfloat32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, dynamic, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64intfloatdyn2d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());
}

} // namespace tf::py

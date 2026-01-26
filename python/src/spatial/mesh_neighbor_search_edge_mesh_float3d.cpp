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
#include <trueform/python/spatial/edge_mesh.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/form_form_neighbor_search.hpp>

namespace tf::py {

auto register_mesh_neighbor_search_edge_mesh_float3d(nanobind::module_ &m) -> void {

  // ==== float, 3D ====

  // int32 mesh, int32 edge_mesh, triangle, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_intint3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, triangle, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_intint643float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, triangle, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int32 edge_mesh, dynamic, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_intintdynfloat3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, dynamic, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_intint64dynfloat3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, dynamic, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, triangle, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64int3float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, dynamic, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64intdynfloat3d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());
}

} // namespace tf::py

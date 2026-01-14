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
#include <trueform/python/spatial/point_cloud.hpp>
#include <trueform/python/spatial/form_form_neighbor_search.hpp>

namespace tf::py {

auto register_edge_mesh_neighbor_search_point_cloud(nanobind::module_ &m)
    -> void {

  // ============================================================================
  // EdgeMesh neighbor_search PointCloud
  // EdgeMesh: 2 index types
  // PointCloud: no index type
  // Real types: float, double (must match)
  // Dims: 2D, 3D (must match)
  // Total: 2 × 2 × 2 = 8 functions
  // ============================================================================

  // int32, float, 2D
  m.def("neighbor_search_edge_mesh_point_cloud_intfloat2d",
        [](edge_mesh_wrapper<int, float, 2> &edge_mesh,
           point_cloud_wrapper<float, 2> &cloud,
           std::optional<float> radius) {
          return form_form_neighbor_search(edge_mesh, cloud, radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("cloud"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32, float, 3D
  m.def("neighbor_search_edge_mesh_point_cloud_intfloat3d",
        [](edge_mesh_wrapper<int, float, 3> &edge_mesh,
           point_cloud_wrapper<float, 3> &cloud,
           std::optional<float> radius) {
          return form_form_neighbor_search(edge_mesh, cloud, radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("cloud"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32, double, 2D
  m.def("neighbor_search_edge_mesh_point_cloud_intdouble2d",
        [](edge_mesh_wrapper<int, double, 2> &edge_mesh,
           point_cloud_wrapper<double, 2> &cloud,
           std::optional<double> radius) {
          return form_form_neighbor_search(edge_mesh, cloud, radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("cloud"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32, double, 3D
  m.def("neighbor_search_edge_mesh_point_cloud_intdouble3d",
        [](edge_mesh_wrapper<int, double, 3> &edge_mesh,
           point_cloud_wrapper<double, 3> &cloud,
           std::optional<double> radius) {
          return form_form_neighbor_search(edge_mesh, cloud, radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("cloud"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64, float, 2D
  m.def("neighbor_search_edge_mesh_point_cloud_int64float2d",
        [](edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           point_cloud_wrapper<float, 2> &cloud,
           std::optional<float> radius) {
          return form_form_neighbor_search(edge_mesh, cloud, radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("cloud"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64, float, 3D
  m.def("neighbor_search_edge_mesh_point_cloud_int64float3d",
        [](edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           point_cloud_wrapper<float, 3> &cloud,
           std::optional<float> radius) {
          return form_form_neighbor_search(edge_mesh, cloud, radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("cloud"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64, double, 2D
  m.def("neighbor_search_edge_mesh_point_cloud_int64double2d",
        [](edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           point_cloud_wrapper<double, 2> &cloud,
           std::optional<double> radius) {
          return form_form_neighbor_search(edge_mesh, cloud, radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("cloud"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64, double, 3D
  m.def("neighbor_search_edge_mesh_point_cloud_int64double3d",
        [](edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           point_cloud_wrapper<double, 3> &cloud,
           std::optional<double> radius) {
          return form_form_neighbor_search(edge_mesh, cloud, radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("cloud"),
        nanobind::arg("radius").none() = nanobind::none());
}

} // namespace tf::py

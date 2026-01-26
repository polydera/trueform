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
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <trueform/python/spatial/edge_mesh.hpp>
#include <trueform/python/core/make_primitives.hpp>
#include <trueform/python/spatial/neighbor_search.hpp>

namespace tf::py {

auto register_edge_mesh_neighbor_search(nanobind::module_ &m) -> void {

  // ============================================================================
  // Non-KNN neighbor search (single nearest neighbor)
  // EdgeMeshWrapperIntFloat2D (int, float, 2D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_edge_mesh_point_intfloat2d",
        [](edge_mesh_wrapper<int, float, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               query,
           std::optional<float> radius) {
          return neighbor_search<float, 2>(
              edge_mesh, make_point_from_array<2, float>(query), radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_edge_mesh_segment_intfloat2d",
      [](edge_mesh_wrapper<int, float, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            edge_mesh, make_segment_from_array<2, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_edge_mesh_polygon_intfloat2d",
      [](edge_mesh_wrapper<int, float, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            edge_mesh, make_polygon_from_array<2, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_edge_mesh_ray_intfloat2d",
      [](edge_mesh_wrapper<int, float, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            edge_mesh, make_ray_from_array<2, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_edge_mesh_line_intfloat2d",
      [](edge_mesh_wrapper<int, float, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            edge_mesh, make_line_from_array<2, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // EdgeMeshWrapperIntFloat3D (int, float, 3D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_edge_mesh_point_intfloat3d",
        [](edge_mesh_wrapper<int, float, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               query,
           std::optional<float> radius) {
          return neighbor_search<float, 3>(
              edge_mesh, make_point_from_array<3, float>(query), radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_edge_mesh_segment_intfloat3d",
      [](edge_mesh_wrapper<int, float, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            edge_mesh, make_segment_from_array<3, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_edge_mesh_polygon_intfloat3d",
      [](edge_mesh_wrapper<int, float, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            edge_mesh, make_polygon_from_array<3, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_edge_mesh_ray_intfloat3d",
      [](edge_mesh_wrapper<int, float, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            edge_mesh, make_ray_from_array<3, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_edge_mesh_line_intfloat3d",
      [](edge_mesh_wrapper<int, float, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            edge_mesh, make_line_from_array<3, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Plane queries
  m.def(
      "neighbor_search_edge_mesh_plane_intfloat3d",
      [](edge_mesh_wrapper<int, float, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            edge_mesh, make_plane_from_array<3, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // EdgeMeshWrapperIntDouble2D (int, double, 2D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_edge_mesh_point_intdouble2d",
        [](edge_mesh_wrapper<int, double, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 2>(
              edge_mesh, make_point_from_array<2, double>(query), radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_edge_mesh_segment_intdouble2d",
      [](edge_mesh_wrapper<int, double, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            edge_mesh, make_segment_from_array<2, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_edge_mesh_polygon_intdouble2d",
      [](edge_mesh_wrapper<int, double, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            edge_mesh, make_polygon_from_array<2, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_edge_mesh_ray_intdouble2d",
      [](edge_mesh_wrapper<int, double, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            edge_mesh, make_ray_from_array<2, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_edge_mesh_line_intdouble2d",
      [](edge_mesh_wrapper<int, double, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            edge_mesh, make_line_from_array<2, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // EdgeMeshWrapperIntDouble3D (int, double, 3D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_edge_mesh_point_intdouble3d",
        [](edge_mesh_wrapper<int, double, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 3>(
              edge_mesh, make_point_from_array<3, double>(query), radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_edge_mesh_segment_intdouble3d",
      [](edge_mesh_wrapper<int, double, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            edge_mesh, make_segment_from_array<3, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_edge_mesh_polygon_intdouble3d",
      [](edge_mesh_wrapper<int, double, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            edge_mesh, make_polygon_from_array<3, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_edge_mesh_ray_intdouble3d",
      [](edge_mesh_wrapper<int, double, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            edge_mesh, make_ray_from_array<3, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_edge_mesh_line_intdouble3d",
      [](edge_mesh_wrapper<int, double, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            edge_mesh, make_line_from_array<3, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Plane queries
  m.def(
      "neighbor_search_edge_mesh_plane_intdouble3d",
      [](edge_mesh_wrapper<int, double, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            edge_mesh, make_plane_from_array<3, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // EdgeMeshWrapperInt64Float2D (int64, float, 2D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_edge_mesh_point_int64float2d",
        [](edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               query,
           std::optional<float> radius) {
          return neighbor_search<float, 2>(
              edge_mesh, make_point_from_array<2, float>(query), radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_edge_mesh_segment_int64float2d",
      [](edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            edge_mesh, make_segment_from_array<2, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_edge_mesh_polygon_int64float2d",
      [](edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            edge_mesh, make_polygon_from_array<2, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_edge_mesh_ray_int64float2d",
      [](edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            edge_mesh, make_ray_from_array<2, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_edge_mesh_line_int64float2d",
      [](edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            edge_mesh, make_line_from_array<2, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // EdgeMeshWrapperInt64Float3D (int64, float, 3D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_edge_mesh_point_int64float3d",
        [](edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               query,
           std::optional<float> radius) {
          return neighbor_search<float, 3>(
              edge_mesh, make_point_from_array<3, float>(query), radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_edge_mesh_segment_int64float3d",
      [](edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            edge_mesh, make_segment_from_array<3, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_edge_mesh_polygon_int64float3d",
      [](edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            edge_mesh, make_polygon_from_array<3, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_edge_mesh_ray_int64float3d",
      [](edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            edge_mesh, make_ray_from_array<3, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_edge_mesh_line_int64float3d",
      [](edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            edge_mesh, make_line_from_array<3, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Plane queries
  m.def(
      "neighbor_search_edge_mesh_plane_int64float3d",
      [](edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            edge_mesh, make_plane_from_array<3, float>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // EdgeMeshWrapperInt64Double2D (int64, double, 2D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_edge_mesh_point_int64double2d",
        [](edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 2>(
              edge_mesh, make_point_from_array<2, double>(query), radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_edge_mesh_segment_int64double2d",
      [](edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            edge_mesh, make_segment_from_array<2, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_edge_mesh_polygon_int64double2d",
      [](edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            edge_mesh, make_polygon_from_array<2, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_edge_mesh_ray_int64double2d",
      [](edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            edge_mesh, make_ray_from_array<2, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_edge_mesh_line_int64double2d",
      [](edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            edge_mesh, make_line_from_array<2, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // EdgeMeshWrapperInt64Double3D (int64, double, 3D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_edge_mesh_point_int64double3d",
        [](edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 3>(
              edge_mesh, make_point_from_array<3, double>(query), radius);
        },
        nanobind::arg("edge_mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_edge_mesh_segment_int64double3d",
      [](edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            edge_mesh, make_segment_from_array<3, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_edge_mesh_polygon_int64double3d",
      [](edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            edge_mesh, make_polygon_from_array<3, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_edge_mesh_ray_int64double3d",
      [](edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            edge_mesh, make_ray_from_array<3, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_edge_mesh_line_int64double3d",
      [](edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            edge_mesh, make_line_from_array<3, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Plane queries
  m.def(
      "neighbor_search_edge_mesh_plane_int64double3d",
      [](edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            edge_mesh, make_plane_from_array<3, double>(query), radius);
      },
      nanobind::arg("edge_mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());
}

} // namespace tf::py

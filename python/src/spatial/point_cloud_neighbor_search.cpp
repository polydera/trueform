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
#include <trueform/python/core/make_primitives.hpp>
#include <trueform/python/spatial/point_cloud.hpp>
#include <trueform/python/spatial/neighbor_search.hpp>

namespace tf::py {

auto register_point_cloud_neighbor_search(nanobind::module_ &m) -> void {

  // ============================================================================
  // Non-KNN neighbor search (single nearest neighbor)
  // ============================================================================

  // Point queries - 2D float
  m.def("neighbor_search_point_float2d",
        [](point_cloud_wrapper<float, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               query,
           std::optional<float> radius) {
          return neighbor_search<float, 2>(
              cloud, make_point_from_array<2, float>(query), radius);
        },
        nanobind::arg("cloud"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Point queries - 2D double
  m.def("neighbor_search_point_double2d",
        [](point_cloud_wrapper<double, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 2>(
              cloud, make_point_from_array<2, double>(query), radius);
        },
        nanobind::arg("cloud"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Point queries - 3D float
  m.def("neighbor_search_point_float3d",
        [](point_cloud_wrapper<float, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               query,
           std::optional<float> radius) {
          return neighbor_search<float, 3>(
              cloud, make_point_from_array<3, float>(query), radius);
        },
        nanobind::arg("cloud"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Point queries - 3D double
  m.def("neighbor_search_point_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 3>(
              cloud, make_point_from_array<3, double>(query), radius);
        },
        nanobind::arg("cloud"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries - 2D float
  m.def(
      "neighbor_search_segment_float2d",
      [](point_cloud_wrapper<float, 2> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            cloud, make_segment_from_array<2, float>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Segment queries - 2D double
  m.def(
      "neighbor_search_segment_double2d",
      [](point_cloud_wrapper<double, 2> &cloud,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            cloud, make_segment_from_array<2, double>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Segment queries - 3D float
  m.def(
      "neighbor_search_segment_float3d",
      [](point_cloud_wrapper<float, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            cloud, make_segment_from_array<3, float>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Segment queries - 3D double
  m.def(
      "neighbor_search_segment_double3d",
      [](point_cloud_wrapper<double, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            cloud, make_segment_from_array<3, double>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries - 2D float
  m.def(
      "neighbor_search_polygon_float2d",
      [](point_cloud_wrapper<float, 2> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            cloud, make_polygon_from_array<2, float>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries - 2D double
  m.def("neighbor_search_polygon_double2d",
        [](point_cloud_wrapper<double, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<-1, 2>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 2>(
              cloud, make_polygon_from_array<2, double>(query), radius);
        },
        nanobind::arg("cloud"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries - 3D float
  m.def(
      "neighbor_search_polygon_float3d",
      [](point_cloud_wrapper<float, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            cloud, make_polygon_from_array<3, float>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries - 3D double
  m.def("neighbor_search_polygon_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<-1, 3>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 3>(
              cloud, make_polygon_from_array<3, double>(query), radius);
        },
        nanobind::arg("cloud"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Ray queries - 2D float
  m.def(
      "neighbor_search_ray_float2d",
      [](point_cloud_wrapper<float, 2> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            cloud, make_ray_from_array<2, float>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries - 2D double
  m.def(
      "neighbor_search_ray_double2d",
      [](point_cloud_wrapper<double, 2> &cloud,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            cloud, make_ray_from_array<2, double>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries - 3D float
  m.def(
      "neighbor_search_ray_float3d",
      [](point_cloud_wrapper<float, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            cloud, make_ray_from_array<3, float>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries - 3D double
  m.def(
      "neighbor_search_ray_double3d",
      [](point_cloud_wrapper<double, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            cloud, make_ray_from_array<3, double>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries - 2D float
  m.def(
      "neighbor_search_line_float2d",
      [](point_cloud_wrapper<float, 2> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            cloud, make_line_from_array<2, float>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries - 2D double
  m.def(
      "neighbor_search_line_double2d",
      [](point_cloud_wrapper<double, 2> &cloud,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            cloud, make_line_from_array<2, double>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries - 3D float
  m.def(
      "neighbor_search_line_float3d",
      [](point_cloud_wrapper<float, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            cloud, make_line_from_array<3, float>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries - 3D double
  m.def(
      "neighbor_search_line_double3d",
      [](point_cloud_wrapper<double, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            cloud, make_line_from_array<3, double>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Plane queries - 3D float
  m.def(
      "neighbor_search_plane_float3d",
      [](point_cloud_wrapper<float, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            cloud, make_plane_from_array<3, float>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Plane queries - 3D double
  m.def(
      "neighbor_search_plane_double3d",
      [](point_cloud_wrapper<double, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            cloud, make_plane_from_array<3, double>(query), radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // KNN neighbor search (k nearest neighbors)
  // ============================================================================

  // Point queries - 2D float
  m.def("neighbor_search_knn_point_float2d",
        [](point_cloud_wrapper<float, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               query,
           int k, std::optional<float> radius) {
          return neighbor_search<float, 2>(
              cloud, make_point_from_array<2, float>(query), k, radius);
        },
        nanobind::arg("cloud"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  // Point queries - 2D double
  m.def("neighbor_search_knn_point_double2d",
        [](point_cloud_wrapper<double, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               query,
           int k, std::optional<double> radius) {
          return neighbor_search<double, 2>(
              cloud, make_point_from_array<2, double>(query), k, radius);
        },
        nanobind::arg("cloud"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  // Point queries - 3D float
  m.def("neighbor_search_knn_point_float3d",
        [](point_cloud_wrapper<float, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               query,
           int k, std::optional<float> radius) {
          return neighbor_search<float, 3>(
              cloud, make_point_from_array<3, float>(query), k, radius);
        },
        nanobind::arg("cloud"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  // Point queries - 3D double
  m.def("neighbor_search_knn_point_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               query,
           int k, std::optional<double> radius) {
          return neighbor_search<double, 3>(
              cloud, make_point_from_array<3, double>(query), k, radius);
        },
        nanobind::arg("cloud"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries - 2D float
  m.def(
      "neighbor_search_knn_segment_float2d",
      [](point_cloud_wrapper<float, 2> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            cloud, make_segment_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Segment queries - 2D double
  m.def(
      "neighbor_search_knn_segment_double2d",
      [](point_cloud_wrapper<double, 2> &cloud,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            cloud, make_segment_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Segment queries - 3D float
  m.def(
      "neighbor_search_knn_segment_float3d",
      [](point_cloud_wrapper<float, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            cloud, make_segment_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Segment queries - 3D double
  m.def(
      "neighbor_search_knn_segment_double3d",
      [](point_cloud_wrapper<double, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            cloud, make_segment_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries - 2D float
  m.def(
      "neighbor_search_knn_polygon_float2d",
      [](point_cloud_wrapper<float, 2> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            cloud, make_polygon_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries - 2D double
  m.def("neighbor_search_knn_polygon_double2d",
        [](point_cloud_wrapper<double, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<-1, 2>>
               query,
           int k, std::optional<double> radius) {
          return neighbor_search<double, 2>(
              cloud, make_polygon_from_array<2, double>(query), k, radius);
        },
        nanobind::arg("cloud"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries - 3D float
  m.def(
      "neighbor_search_knn_polygon_float3d",
      [](point_cloud_wrapper<float, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            cloud, make_polygon_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries - 3D double
  m.def("neighbor_search_knn_polygon_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<-1, 3>>
               query,
           int k, std::optional<double> radius) {
          return neighbor_search<double, 3>(
              cloud, make_polygon_from_array<3, double>(query), k, radius);
        },
        nanobind::arg("cloud"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  // Ray queries - 2D float
  m.def(
      "neighbor_search_knn_ray_float2d",
      [](point_cloud_wrapper<float, 2> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            cloud, make_ray_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries - 2D double
  m.def(
      "neighbor_search_knn_ray_double2d",
      [](point_cloud_wrapper<double, 2> &cloud,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            cloud, make_ray_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries - 3D float
  m.def(
      "neighbor_search_knn_ray_float3d",
      [](point_cloud_wrapper<float, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            cloud, make_ray_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries - 3D double
  m.def(
      "neighbor_search_knn_ray_double3d",
      [](point_cloud_wrapper<double, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            cloud, make_ray_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries - 2D float
  m.def(
      "neighbor_search_knn_line_float2d",
      [](point_cloud_wrapper<float, 2> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            cloud, make_line_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries - 2D double
  m.def(
      "neighbor_search_knn_line_double2d",
      [](point_cloud_wrapper<double, 2> &cloud,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            cloud, make_line_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries - 3D float
  m.def(
      "neighbor_search_knn_line_float3d",
      [](point_cloud_wrapper<float, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            cloud, make_line_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries - 3D double
  m.def(
      "neighbor_search_knn_line_double3d",
      [](point_cloud_wrapper<double, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            cloud, make_line_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Plane queries - 3D float
  m.def(
      "neighbor_search_knn_plane_float3d",
      [](point_cloud_wrapper<float, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            cloud, make_plane_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // Plane queries - 3D double
  m.def(
      "neighbor_search_knn_plane_double3d",
      [](point_cloud_wrapper<double, 3> &cloud,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            cloud, make_plane_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("cloud"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());
}

} // namespace tf::py

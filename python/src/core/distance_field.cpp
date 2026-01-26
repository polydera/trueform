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

#include "trueform/python/core/distance_field.hpp"
#include "trueform/python/core/make_primitives.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_core_distance_field(nanobind::module_ &m) -> void {
  // ==== Distance field to Plane (3D only) ====

  // float, 3D
  m.def(
      "distance_field_plane_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             points,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
             plane_data) {
        auto plane = make_plane_from_array<3, float>(plane_data);
        return distance_field_impl<3, float>(points, plane);
      },
      nanobind::arg("points"), nanobind::arg("plane"));

  // double, 3D
  m.def(
      "distance_field_plane_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1, 3>>
             points,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
             plane_data) {
        auto plane = make_plane_from_array<3, double>(plane_data);
        return distance_field_impl<3, double>(points, plane);
      },
      nanobind::arg("points"), nanobind::arg("plane"));

  // ==== Distance field to Segment ====

  // float, 2D
  m.def(
      "distance_field_segment_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             points,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             segment_data) {
        auto segment = make_segment_from_array<2, float>(segment_data);
        return distance_field_impl<2, float>(points, segment);
      },
      nanobind::arg("points"), nanobind::arg("segment"));

  // float, 3D
  m.def(
      "distance_field_segment_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             points,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             segment_data) {
        auto segment = make_segment_from_array<3, float>(segment_data);
        return distance_field_impl<3, float>(points, segment);
      },
      nanobind::arg("points"), nanobind::arg("segment"));

  // double, 2D
  m.def(
      "distance_field_segment_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1, 2>>
             points,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             segment_data) {
        auto segment = make_segment_from_array<2, double>(segment_data);
        return distance_field_impl<2, double>(points, segment);
      },
      nanobind::arg("points"), nanobind::arg("segment"));

  // double, 3D
  m.def(
      "distance_field_segment_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1, 3>>
             points,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             segment_data) {
        auto segment = make_segment_from_array<3, double>(segment_data);
        return distance_field_impl<3, double>(points, segment);
      },
      nanobind::arg("points"), nanobind::arg("segment"));

  // ==== Distance field to Polygon ====

  // float, 2D
  m.def(
      "distance_field_polygon_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             points,
         nanobind::ndarray<nanobind::numpy, const float> poly_data) {
        auto polygon = make_polygon_from_array<2, float>(poly_data);
        return distance_field_impl<2, float>(points, polygon);
      },
      nanobind::arg("points"), nanobind::arg("polygon"));

  // float, 3D
  m.def(
      "distance_field_polygon_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             points,
         nanobind::ndarray<nanobind::numpy, const float> poly_data) {
        auto polygon = make_polygon_from_array<3, float>(poly_data);
        return distance_field_impl<3, float>(points, polygon);
      },
      nanobind::arg("points"), nanobind::arg("polygon"));

  // double, 2D
  m.def(
      "distance_field_polygon_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1, 2>>
             points,
         nanobind::ndarray<nanobind::numpy, const double> poly_data) {
        auto polygon = make_polygon_from_array<2, double>(poly_data);
        return distance_field_impl<2, double>(points, polygon);
      },
      nanobind::arg("points"), nanobind::arg("polygon"));

  // double, 3D
  m.def(
      "distance_field_polygon_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1, 3>>
             points,
         nanobind::ndarray<nanobind::numpy, const double> poly_data) {
        auto polygon = make_polygon_from_array<3, double>(poly_data);
        return distance_field_impl<3, double>(points, polygon);
      },
      nanobind::arg("points"), nanobind::arg("polygon"));

  // ==== Distance field to Line ====

  // float, 2D
  m.def(
      "distance_field_line_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             points,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             line_data) {
        auto line = make_line_from_array<2, float>(line_data);
        return distance_field_impl<2, float>(points, line);
      },
      nanobind::arg("points"), nanobind::arg("line"));

  // float, 3D
  m.def(
      "distance_field_line_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             points,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             line_data) {
        auto line = make_line_from_array<3, float>(line_data);
        return distance_field_impl<3, float>(points, line);
      },
      nanobind::arg("points"), nanobind::arg("line"));

  // double, 2D
  m.def(
      "distance_field_line_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1, 2>>
             points,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             line_data) {
        auto line = make_line_from_array<2, double>(line_data);
        return distance_field_impl<2, double>(points, line);
      },
      nanobind::arg("points"), nanobind::arg("line"));

  // double, 3D
  m.def(
      "distance_field_line_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1, 3>>
             points,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             line_data) {
        auto line = make_line_from_array<3, double>(line_data);
        return distance_field_impl<3, double>(points, line);
      },
      nanobind::arg("points"), nanobind::arg("line"));

  // ==== Distance field to AABB ====

  // float, 2D
  m.def(
      "distance_field_aabb_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             points,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             aabb_data) {
        auto aabb = make_aabb_from_array<2, float>(aabb_data);
        return distance_field_impl<2, float>(points, aabb);
      },
      nanobind::arg("points"), nanobind::arg("aabb"));

  // float, 3D
  m.def(
      "distance_field_aabb_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             points,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             aabb_data) {
        auto aabb = make_aabb_from_array<3, float>(aabb_data);
        return distance_field_impl<3, float>(points, aabb);
      },
      nanobind::arg("points"), nanobind::arg("aabb"));

  // double, 2D
  m.def(
      "distance_field_aabb_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1, 2>>
             points,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             aabb_data) {
        auto aabb = make_aabb_from_array<2, double>(aabb_data);
        return distance_field_impl<2, double>(points, aabb);
      },
      nanobind::arg("points"), nanobind::arg("aabb"));

  // double, 3D
  m.def(
      "distance_field_aabb_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1, 3>>
             points,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             aabb_data) {
        auto aabb = make_aabb_from_array<3, double>(aabb_data);
        return distance_field_impl<3, double>(points, aabb);
      },
      nanobind::arg("points"), nanobind::arg("aabb"));
}

} // namespace tf::py

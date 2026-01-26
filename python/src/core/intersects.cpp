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

#include "trueform/python/core/intersects.hpp"
#include "trueform/python/core/make_primitives.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/intersects.hpp>

namespace tf::py {

auto register_core_intersects(nanobind::module_ &m) -> void {
  // ==== Point to Point ====
  m.def("intersects_point_point_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt1_data) {
          auto pt0 = make_point_from_array<2, float>(pt0_data);
          auto pt1 = make_point_from_array<2, float>(pt1_data);
          return tf::intersects(pt0, pt1);
        });

  m.def("intersects_point_point_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt1_data) {
          auto pt0 = make_point_from_array<3, float>(pt0_data);
          auto pt1 = make_point_from_array<3, float>(pt1_data);
          return tf::intersects(pt0, pt1);
        });

  m.def("intersects_point_point_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt0_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt1_data) {
          auto pt0 = make_point_from_array<2, double>(pt0_data);
          auto pt1 = make_point_from_array<2, double>(pt1_data);
          return tf::intersects(pt0, pt1);
        });

  m.def("intersects_point_point_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt0_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt1_data) {
          auto pt0 = make_point_from_array<3, double>(pt0_data);
          auto pt1 = make_point_from_array<3, double>(pt1_data);
          return tf::intersects(pt0, pt1);
        });

  // ==== Point to AABB ====
  m.def("intersects_point_aabb_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               aabb_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto aabb = make_aabb_from_array<2, float>(aabb_data);
          return tf::intersects(pt, aabb);
        });

  m.def("intersects_point_aabb_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               aabb_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto aabb = make_aabb_from_array<3, float>(aabb_data);
          return tf::intersects(pt, aabb);
        });

  m.def("intersects_point_aabb_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               aabb_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          auto aabb = make_aabb_from_array<2, double>(aabb_data);
          return tf::intersects(pt, aabb);
        });

  m.def("intersects_point_aabb_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               aabb_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          auto aabb = make_aabb_from_array<3, double>(aabb_data);
          return tf::intersects(pt, aabb);
        });

  // ==== Point to Line ====
  m.def("intersects_point_line_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto line = make_line_from_array<2, float>(line_data);
          return tf::intersects(pt, line);
        });

  m.def("intersects_point_line_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto line = make_line_from_array<3, float>(line_data);
          return tf::intersects(pt, line);
        });

  m.def("intersects_point_line_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               line_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          auto line = make_line_from_array<2, double>(line_data);
          return tf::intersects(pt, line);
        });

  m.def("intersects_point_line_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               line_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          auto line = make_line_from_array<3, double>(line_data);
          return tf::intersects(pt, line);
        });

  // ==== Point to Ray ====
  m.def("intersects_point_ray_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto ray = make_ray_from_array<2, float>(ray_data);
          return tf::intersects(pt, ray);
        });

  m.def("intersects_point_ray_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto ray = make_ray_from_array<3, float>(ray_data);
          return tf::intersects(pt, ray);
        });

  m.def("intersects_point_ray_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               ray_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          auto ray = make_ray_from_array<2, double>(ray_data);
          return tf::intersects(pt, ray);
        });

  m.def("intersects_point_ray_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               ray_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          auto ray = make_ray_from_array<3, double>(ray_data);
          return tf::intersects(pt, ray);
        });

  // ==== Point to Segment ====
  m.def("intersects_point_segment_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto seg = make_segment_from_array<2, float>(seg_data);
          return tf::intersects(pt, seg);
        });

  m.def("intersects_point_segment_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto seg = make_segment_from_array<3, float>(seg_data);
          return tf::intersects(pt, seg);
        });

  m.def("intersects_point_segment_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               seg_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          auto seg = make_segment_from_array<2, double>(seg_data);
          return tf::intersects(pt, seg);
        });

  m.def("intersects_point_segment_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               seg_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          auto seg = make_segment_from_array<3, double>(seg_data);
          return tf::intersects(pt, seg);
        });

  // ==== Point to Polygon ====
  m.def("intersects_point_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return tf::intersects(pt, poly);
        });

  m.def("intersects_point_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return tf::intersects(pt, poly);
        });

  m.def("intersects_point_polygon_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return tf::intersects(pt, poly);
        });

  m.def("intersects_point_polygon_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return tf::intersects(pt, poly);
        });

  // ==== Point to Plane (3D only) ====
  m.def("intersects_point_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::intersects(pt, plane);
        });

  m.def("intersects_point_plane_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          auto plane = make_plane_from_array<3, double>(plane_data);
          return tf::intersects(pt, plane);
        });

  // ==== AABB to AABB ====
  m.def("intersects_aabb_aabb_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               aabb0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               aabb1_data) {
          auto aabb0 = make_aabb_from_array<2, float>(aabb0_data);
          auto aabb1 = make_aabb_from_array<2, float>(aabb1_data);
          return tf::intersects(aabb0, aabb1);
        });

  m.def("intersects_aabb_aabb_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               aabb0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               aabb1_data) {
          auto aabb0 = make_aabb_from_array<3, float>(aabb0_data);
          auto aabb1 = make_aabb_from_array<3, float>(aabb1_data);
          return tf::intersects(aabb0, aabb1);
        });

  m.def("intersects_aabb_aabb_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               aabb0_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               aabb1_data) {
          auto aabb0 = make_aabb_from_array<2, double>(aabb0_data);
          auto aabb1 = make_aabb_from_array<2, double>(aabb1_data);
          return tf::intersects(aabb0, aabb1);
        });

  m.def("intersects_aabb_aabb_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               aabb0_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               aabb1_data) {
          auto aabb0 = make_aabb_from_array<3, double>(aabb0_data);
          auto aabb1 = make_aabb_from_array<3, double>(aabb1_data);
          return tf::intersects(aabb0, aabb1);
        });

  // ==== AABB to Plane (3D only) ====
  m.def("intersects_aabb_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               aabb_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto aabb = make_aabb_from_array<3, float>(aabb_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::intersects(aabb, plane);
        });

  m.def("intersects_aabb_plane_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               aabb_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto aabb = make_aabb_from_array<3, double>(aabb_data);
          auto plane = make_plane_from_array<3, double>(plane_data);
          return tf::intersects(aabb, plane);
        });

  // ==== Line to Line ====
  m.def("intersects_line_line_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line1_data) {
          auto line0 = make_line_from_array<2, float>(line0_data);
          auto line1 = make_line_from_array<2, float>(line1_data);
          return tf::intersects(line0, line1);
        });

  m.def("intersects_line_line_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line1_data) {
          auto line0 = make_line_from_array<3, float>(line0_data);
          auto line1 = make_line_from_array<3, float>(line1_data);
          return tf::intersects(line0, line1);
        });

  m.def("intersects_line_line_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               line0_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               line1_data) {
          auto line0 = make_line_from_array<2, double>(line0_data);
          auto line1 = make_line_from_array<2, double>(line1_data);
          return tf::intersects(line0, line1);
        });

  m.def("intersects_line_line_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               line0_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               line1_data) {
          auto line0 = make_line_from_array<3, double>(line0_data);
          auto line1 = make_line_from_array<3, double>(line1_data);
          return tf::intersects(line0, line1);
        });

  // ==== Line to Ray ====
  m.def("intersects_line_ray_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto line = make_line_from_array<2, float>(line_data);
          auto ray = make_ray_from_array<2, float>(ray_data);
          return tf::intersects(line, ray);
        });

  m.def("intersects_line_ray_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto line = make_line_from_array<3, float>(line_data);
          auto ray = make_ray_from_array<3, float>(ray_data);
          return tf::intersects(line, ray);
        });

  m.def("intersects_line_ray_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               ray_data) {
          auto line = make_line_from_array<2, double>(line_data);
          auto ray = make_ray_from_array<2, double>(ray_data);
          return tf::intersects(line, ray);
        });

  m.def("intersects_line_ray_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               ray_data) {
          auto line = make_line_from_array<3, double>(line_data);
          auto ray = make_ray_from_array<3, double>(ray_data);
          return tf::intersects(line, ray);
        });

  // ==== Line to Segment ====
  m.def("intersects_line_segment_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto line = make_line_from_array<2, float>(line_data);
          auto seg = make_segment_from_array<2, float>(seg_data);
          return tf::intersects(line, seg);
        });

  m.def("intersects_line_segment_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto line = make_line_from_array<3, float>(line_data);
          auto seg = make_segment_from_array<3, float>(seg_data);
          return tf::intersects(line, seg);
        });

  m.def("intersects_line_segment_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               seg_data) {
          auto line = make_line_from_array<2, double>(line_data);
          auto seg = make_segment_from_array<2, double>(seg_data);
          return tf::intersects(line, seg);
        });

  m.def("intersects_line_segment_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               seg_data) {
          auto line = make_line_from_array<3, double>(line_data);
          auto seg = make_segment_from_array<3, double>(seg_data);
          return tf::intersects(line, seg);
        });

  // ==== Line to Polygon ====
  m.def("intersects_line_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto line = make_line_from_array<2, float>(line_data);
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return tf::intersects(line, poly);
        });

  m.def("intersects_line_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto line = make_line_from_array<3, float>(line_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return tf::intersects(line, poly);
        });

  m.def("intersects_line_polygon_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto line = make_line_from_array<2, double>(line_data);
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return tf::intersects(line, poly);
        });

  m.def("intersects_line_polygon_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto line = make_line_from_array<3, double>(line_data);
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return tf::intersects(line, poly);
        });

  // ==== Line to Plane (3D only) ====
  m.def("intersects_line_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto line = make_line_from_array<3, float>(line_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::intersects(line, plane);
        });

  m.def("intersects_line_plane_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto line = make_line_from_array<3, double>(line_data);
          auto plane = make_plane_from_array<3, double>(plane_data);
          return tf::intersects(line, plane);
        });

  // ==== Ray to Ray ====
  m.def("intersects_ray_ray_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray1_data) {
          auto ray0 = make_ray_from_array<2, float>(ray0_data);
          auto ray1 = make_ray_from_array<2, float>(ray1_data);
          return tf::intersects(ray0, ray1);
        });

  m.def("intersects_ray_ray_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray1_data) {
          auto ray0 = make_ray_from_array<3, float>(ray0_data);
          auto ray1 = make_ray_from_array<3, float>(ray1_data);
          return tf::intersects(ray0, ray1);
        });

  m.def("intersects_ray_ray_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               ray0_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               ray1_data) {
          auto ray0 = make_ray_from_array<2, double>(ray0_data);
          auto ray1 = make_ray_from_array<2, double>(ray1_data);
          return tf::intersects(ray0, ray1);
        });

  m.def("intersects_ray_ray_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               ray0_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               ray1_data) {
          auto ray0 = make_ray_from_array<3, double>(ray0_data);
          auto ray1 = make_ray_from_array<3, double>(ray1_data);
          return tf::intersects(ray0, ray1);
        });

  // ==== Ray to Segment ====
  m.def("intersects_ray_segment_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          auto seg = make_segment_from_array<2, float>(seg_data);
          return tf::intersects(ray, seg);
        });

  m.def("intersects_ray_segment_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          auto seg = make_segment_from_array<3, float>(seg_data);
          return tf::intersects(ray, seg);
        });

  m.def("intersects_ray_segment_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               seg_data) {
          auto ray = make_ray_from_array<2, double>(ray_data);
          auto seg = make_segment_from_array<2, double>(seg_data);
          return tf::intersects(ray, seg);
        });

  m.def("intersects_ray_segment_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               seg_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          auto seg = make_segment_from_array<3, double>(seg_data);
          return tf::intersects(ray, seg);
        });

  // ==== Ray to Polygon ====
  m.def("intersects_ray_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return tf::intersects(ray, poly);
        });

  m.def("intersects_ray_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return tf::intersects(ray, poly);
        });

  m.def("intersects_ray_polygon_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto ray = make_ray_from_array<2, double>(ray_data);
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return tf::intersects(ray, poly);
        });

  m.def("intersects_ray_polygon_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return tf::intersects(ray, poly);
        });

  // ==== Ray to Plane (3D only) ====
  m.def("intersects_ray_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::intersects(ray, plane);
        });

  m.def("intersects_ray_plane_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          auto plane = make_plane_from_array<3, double>(plane_data);
          return tf::intersects(ray, plane);
        });

  // ==== Segment to Segment ====
  m.def("intersects_segment_segment_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg1_data) {
          auto seg0 = make_segment_from_array<2, float>(seg0_data);
          auto seg1 = make_segment_from_array<2, float>(seg1_data);
          return tf::intersects(seg0, seg1);
        });

  m.def("intersects_segment_segment_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg1_data) {
          auto seg0 = make_segment_from_array<3, float>(seg0_data);
          auto seg1 = make_segment_from_array<3, float>(seg1_data);
          return tf::intersects(seg0, seg1);
        });

  m.def("intersects_segment_segment_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               seg0_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               seg1_data) {
          auto seg0 = make_segment_from_array<2, double>(seg0_data);
          auto seg1 = make_segment_from_array<2, double>(seg1_data);
          return tf::intersects(seg0, seg1);
        });

  m.def("intersects_segment_segment_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               seg0_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               seg1_data) {
          auto seg0 = make_segment_from_array<3, double>(seg0_data);
          auto seg1 = make_segment_from_array<3, double>(seg1_data);
          return tf::intersects(seg0, seg1);
        });

  // ==== Segment to Polygon ====
  m.def("intersects_segment_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return tf::intersects(seg, poly);
        });

  m.def("intersects_segment_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return tf::intersects(seg, poly);
        });

  m.def("intersects_segment_polygon_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto seg = make_segment_from_array<2, double>(seg_data);
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return tf::intersects(seg, poly);
        });

  m.def("intersects_segment_polygon_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto seg = make_segment_from_array<3, double>(seg_data);
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return tf::intersects(seg, poly);
        });

  // ==== Segment to Plane (3D only) ====
  m.def("intersects_segment_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::intersects(seg, plane);
        });

  m.def("intersects_segment_plane_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto seg = make_segment_from_array<3, double>(seg_data);
          auto plane = make_plane_from_array<3, double>(plane_data);
          return tf::intersects(seg, plane);
        });

  // ==== Polygon to Polygon ====
  m.def("intersects_polygon_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float> poly0_data,
           nanobind::ndarray<nanobind::numpy, const float> poly1_data) {
          auto poly0 = make_polygon_from_array<2, float>(poly0_data);
          auto poly1 = make_polygon_from_array<2, float>(poly1_data);
          return tf::intersects(poly0, poly1);
        });

  m.def("intersects_polygon_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float> poly0_data,
           nanobind::ndarray<nanobind::numpy, const float> poly1_data) {
          auto poly0 = make_polygon_from_array<3, float>(poly0_data);
          auto poly1 = make_polygon_from_array<3, float>(poly1_data);
          return tf::intersects(poly0, poly1);
        });

  m.def("intersects_polygon_polygon_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double> poly0_data,
           nanobind::ndarray<nanobind::numpy, const double> poly1_data) {
          auto poly0 = make_polygon_from_array<2, double>(poly0_data);
          auto poly1 = make_polygon_from_array<2, double>(poly1_data);
          return tf::intersects(poly0, poly1);
        });

  m.def("intersects_polygon_polygon_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double> poly0_data,
           nanobind::ndarray<nanobind::numpy, const double> poly1_data) {
          auto poly0 = make_polygon_from_array<3, double>(poly0_data);
          auto poly1 = make_polygon_from_array<3, double>(poly1_data);
          return tf::intersects(poly0, poly1);
        });

  // ==== Polygon to Plane (3D only) ====
  m.def("intersects_polygon_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float> poly_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::intersects(poly, plane);
        });

  m.def("intersects_polygon_plane_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double> poly_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto poly = make_polygon_from_array<3, double>(poly_data);
          auto plane = make_plane_from_array<3, double>(plane_data);
          return tf::intersects(poly, plane);
        });

  // ==== Segment to AABB ====
  m.def("intersects_segment_aabb_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               aabb_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          auto aabb = make_aabb_from_array<2, float>(aabb_data);
          return tf::intersects(seg, aabb);
        });

  m.def("intersects_segment_aabb_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               aabb_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          auto aabb = make_aabb_from_array<3, float>(aabb_data);
          return tf::intersects(seg, aabb);
        });

  m.def("intersects_segment_aabb_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               aabb_data) {
          auto seg = make_segment_from_array<2, double>(seg_data);
          auto aabb = make_aabb_from_array<2, double>(aabb_data);
          return tf::intersects(seg, aabb);
        });

  m.def("intersects_segment_aabb_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               aabb_data) {
          auto seg = make_segment_from_array<3, double>(seg_data);
          auto aabb = make_aabb_from_array<3, double>(aabb_data);
          return tf::intersects(seg, aabb);
        });

  // ==== Ray to AABB ====
  m.def("intersects_ray_aabb_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               aabb_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          auto aabb = make_aabb_from_array<2, float>(aabb_data);
          return tf::intersects(ray, aabb);
        });

  m.def("intersects_ray_aabb_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               aabb_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          auto aabb = make_aabb_from_array<3, float>(aabb_data);
          return tf::intersects(ray, aabb);
        });

  m.def("intersects_ray_aabb_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               aabb_data) {
          auto ray = make_ray_from_array<2, double>(ray_data);
          auto aabb = make_aabb_from_array<2, double>(aabb_data);
          return tf::intersects(ray, aabb);
        });

  m.def("intersects_ray_aabb_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               aabb_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          auto aabb = make_aabb_from_array<3, double>(aabb_data);
          return tf::intersects(ray, aabb);
        });

  // ==== Line to AABB ====
  m.def("intersects_line_aabb_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               aabb_data) {
          auto line = make_line_from_array<2, float>(line_data);
          auto aabb = make_aabb_from_array<2, float>(aabb_data);
          return tf::intersects(line, aabb);
        });

  m.def("intersects_line_aabb_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               aabb_data) {
          auto line = make_line_from_array<3, float>(line_data);
          auto aabb = make_aabb_from_array<3, float>(aabb_data);
          return tf::intersects(line, aabb);
        });

  m.def("intersects_line_aabb_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               aabb_data) {
          auto line = make_line_from_array<2, double>(line_data);
          auto aabb = make_aabb_from_array<2, double>(aabb_data);
          return tf::intersects(line, aabb);
        });

  m.def("intersects_line_aabb_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               aabb_data) {
          auto line = make_line_from_array<3, double>(line_data);
          auto aabb = make_aabb_from_array<3, double>(aabb_data);
          return tf::intersects(line, aabb);
        });

  // ==== Polygon to AABB ====
  m.def("intersects_polygon_aabb_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float> poly_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               aabb_data) {
          auto poly = make_polygon_from_array<2, float>(poly_data);
          auto aabb = make_aabb_from_array<2, float>(aabb_data);
          return tf::intersects(poly, aabb);
        });

  m.def("intersects_polygon_aabb_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float> poly_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               aabb_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          auto aabb = make_aabb_from_array<3, float>(aabb_data);
          return tf::intersects(poly, aabb);
        });

  m.def("intersects_polygon_aabb_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double> poly_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               aabb_data) {
          auto poly = make_polygon_from_array<2, double>(poly_data);
          auto aabb = make_aabb_from_array<2, double>(aabb_data);
          return tf::intersects(poly, aabb);
        });

  m.def("intersects_polygon_aabb_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double> poly_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               aabb_data) {
          auto poly = make_polygon_from_array<3, double>(poly_data);
          auto aabb = make_aabb_from_array<3, double>(aabb_data);
          return tf::intersects(poly, aabb);
        });

  // ==== Plane to Plane (3D only) ====
  m.def("intersects_plane_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane1_data) {
          auto plane0 = make_plane_from_array<3, float>(plane0_data);
          auto plane1 = make_plane_from_array<3, float>(plane1_data);
          return tf::intersects(plane0, plane1);
        });

  m.def("intersects_plane_plane_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane0_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane1_data) {
          auto plane0 = make_plane_from_array<3, double>(plane0_data);
          auto plane1 = make_plane_from_array<3, double>(plane1_data);
          return tf::intersects(plane0, plane1);
        });
}

} // namespace tf::py

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

#include "trueform/python/core/distance.hpp"
#include "trueform/python/core/make_primitives.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/distance.hpp>

namespace tf::py {

auto register_core_distance_float3d(nanobind::module_ &m) -> void {
  // ==== Point to Point ====

  m.def("distance_point_point_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt1_data) {
          auto pt0 = make_point_from_array<3, float>(pt0_data);
          auto pt1 = make_point_from_array<3, float>(pt1_data);
          return tf::distance(pt0, pt1);
        });

  m.def("distance2_point_point_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt1_data) {
          auto pt0 = make_point_from_array<3, float>(pt0_data);
          auto pt1 = make_point_from_array<3, float>(pt1_data);
          return tf::distance2(pt0, pt1);
        });

  m.def("distance_point_aabb_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               aabb_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto aabb = make_aabb_from_array<3, float>(aabb_data);
          return tf::distance(pt, aabb);
        });

  m.def("distance2_point_aabb_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               aabb_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto aabb = make_aabb_from_array<3, float>(aabb_data);
          return tf::distance2(pt, aabb);
        });

  m.def("distance_point_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::distance(pt, plane);
        });

  m.def("distance2_point_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::distance2(pt, plane);
        });

  m.def("distance_point_line_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto line = make_line_from_array<3, float>(line_data);
          return tf::distance(pt, line);
        });

  m.def("distance2_point_line_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto line = make_line_from_array<3, float>(line_data);
          return tf::distance2(pt, line);
        });

  m.def("distance_point_ray_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto ray = make_ray_from_array<3, float>(ray_data);
          return tf::distance(pt, ray);
        });

  m.def("distance2_point_ray_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto ray = make_ray_from_array<3, float>(ray_data);
          return tf::distance2(pt, ray);
        });

  m.def("distance_point_segment_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto seg = make_segment_from_array<3, float>(seg_data);
          return tf::distance(pt, seg);
        });

  m.def("distance2_point_segment_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto seg = make_segment_from_array<3, float>(seg_data);
          return tf::distance2(pt, seg);
        });

  m.def("distance_point_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 3>>
               poly_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return tf::distance(pt, poly);
        });

  m.def("distance2_point_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 3>>
               poly_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return tf::distance2(pt, poly);
        });

  m.def("distance_aabb_aabb_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               aabb0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               aabb1_data) {
          auto aabb0 = make_aabb_from_array<3, float>(aabb0_data);
          auto aabb1 = make_aabb_from_array<3, float>(aabb1_data);
          return tf::distance(aabb0, aabb1);
        });

  m.def("distance2_aabb_aabb_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               aabb0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               aabb1_data) {
          auto aabb0 = make_aabb_from_array<3, float>(aabb0_data);
          auto aabb1 = make_aabb_from_array<3, float>(aabb1_data);
          return tf::distance2(aabb0, aabb1);
        });

  m.def("distance_line_line_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line1_data) {
          auto line0 = make_line_from_array<3, float>(line0_data);
          auto line1 = make_line_from_array<3, float>(line1_data);
          return tf::distance(line0, line1);
        });

  m.def("distance2_line_line_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line1_data) {
          auto line0 = make_line_from_array<3, float>(line0_data);
          auto line1 = make_line_from_array<3, float>(line1_data);
          return tf::distance2(line0, line1);
        });

  m.def("distance_line_ray_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto line = make_line_from_array<3, float>(line_data);
          auto ray = make_ray_from_array<3, float>(ray_data);
          return tf::distance(line, ray);
        });

  m.def("distance2_line_ray_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto line = make_line_from_array<3, float>(line_data);
          auto ray = make_ray_from_array<3, float>(ray_data);
          return tf::distance2(line, ray);
        });

  m.def("distance_line_segment_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto line = make_line_from_array<3, float>(line_data);
          auto seg = make_segment_from_array<3, float>(seg_data);
          return tf::distance(line, seg);
        });

  m.def("distance2_line_segment_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto line = make_line_from_array<3, float>(line_data);
          auto seg = make_segment_from_array<3, float>(seg_data);
          return tf::distance2(line, seg);
        });

  m.def("distance_line_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 3>>
               poly_data) {
          auto line = make_line_from_array<3, float>(line_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return tf::distance(line, poly);
        });

  m.def("distance2_line_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 3>>
               poly_data) {
          auto line = make_line_from_array<3, float>(line_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return tf::distance2(line, poly);
        });

  m.def("distance_ray_ray_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray1_data) {
          auto ray0 = make_ray_from_array<3, float>(ray0_data);
          auto ray1 = make_ray_from_array<3, float>(ray1_data);
          return tf::distance(ray0, ray1);
        });

  m.def("distance2_ray_ray_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray1_data) {
          auto ray0 = make_ray_from_array<3, float>(ray0_data);
          auto ray1 = make_ray_from_array<3, float>(ray1_data);
          return tf::distance2(ray0, ray1);
        });

  m.def("distance_ray_segment_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          auto seg = make_segment_from_array<3, float>(seg_data);
          return tf::distance(ray, seg);
        });

  m.def("distance2_ray_segment_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          auto seg = make_segment_from_array<3, float>(seg_data);
          return tf::distance2(ray, seg);
        });

  m.def("distance_ray_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 3>>
               poly_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return tf::distance(ray, poly);
        });

  m.def("distance2_ray_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 3>>
               poly_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return tf::distance2(ray, poly);
        });

  m.def("distance_segment_segment_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg1_data) {
          auto seg0 = make_segment_from_array<3, float>(seg0_data);
          auto seg1 = make_segment_from_array<3, float>(seg1_data);
          return tf::distance(seg0, seg1);
        });

  m.def("distance2_segment_segment_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg1_data) {
          auto seg0 = make_segment_from_array<3, float>(seg0_data);
          auto seg1 = make_segment_from_array<3, float>(seg1_data);
          return tf::distance2(seg0, seg1);
        });

  m.def("distance_segment_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 3>>
               poly_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return tf::distance(seg, poly);
        });

  m.def("distance2_segment_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 3>>
               poly_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return tf::distance2(seg, poly);
        });

  m.def("distance_polygon_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 3>>
               poly0_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 3>>
               poly1_data) {
          auto poly0 = make_polygon_from_array<3, float>(poly0_data);
          auto poly1 = make_polygon_from_array<3, float>(poly1_data);
          return tf::distance(poly0, poly1);
        });

  m.def("distance2_polygon_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 3>>
               poly0_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 3>>
               poly1_data) {
          auto poly0 = make_polygon_from_array<3, float>(poly0_data);
          auto poly1 = make_polygon_from_array<3, float>(poly1_data);
          return tf::distance2(poly0, poly1);
        });

  // ==== Segment to Plane (3D only) ====

  m.def("distance_segment_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::distance(seg, plane);
        });

  m.def("distance2_segment_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::distance2(seg, plane);
        });

  // ==== Ray to Plane (3D only) ====

  m.def("distance_ray_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::distance(ray, plane);
        });

  m.def("distance2_ray_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::distance2(ray, plane);
        });

  // ==== Line to Plane (3D only) ====

  m.def("distance_line_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto line = make_line_from_array<3, float>(line_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::distance(line, plane);
        });

  m.def("distance2_line_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto line = make_line_from_array<3, float>(line_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::distance2(line, plane);
        });

  // ==== Polygon to Plane (3D only) ====

  m.def("distance_polygon_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
               poly_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::distance(poly, plane);
        });

  m.def("distance2_polygon_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
               poly_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return tf::distance2(poly, plane);
        });

  // ==== Plane to Plane (3D only) ====

  m.def("distance_plane_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane1_data) {
          auto plane0 = make_plane_from_array<3, float>(plane0_data);
          auto plane1 = make_plane_from_array<3, float>(plane1_data);
          return tf::distance(plane0, plane1);
        });

  m.def("distance2_plane_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane1_data) {
          auto plane0 = make_plane_from_array<3, float>(plane0_data);
          auto plane1 = make_plane_from_array<3, float>(plane1_data);
          return tf::distance2(plane0, plane1);
        });

}

} // namespace tf::py

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

auto register_core_distance_float2d(nanobind::module_ &m) -> void {
  // ==== Point to Point ====

  m.def("distance_point_point_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt1_data) {
          auto pt0 = make_point_from_array<2, float>(pt0_data);
          auto pt1 = make_point_from_array<2, float>(pt1_data);
          return tf::distance(pt0, pt1);
        });

  m.def("distance2_point_point_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt1_data) {
          auto pt0 = make_point_from_array<2, float>(pt0_data);
          auto pt1 = make_point_from_array<2, float>(pt1_data);
          return tf::distance2(pt0, pt1);
        });

  m.def("distance_point_aabb_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               aabb_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto aabb = make_aabb_from_array<2, float>(aabb_data);
          return tf::distance(pt, aabb);
        });

  m.def("distance2_point_aabb_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               aabb_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto aabb = make_aabb_from_array<2, float>(aabb_data);
          return tf::distance2(pt, aabb);
        });

  m.def("distance_point_line_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto line = make_line_from_array<2, float>(line_data);
          return tf::distance(pt, line);
        });

  m.def("distance2_point_line_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto line = make_line_from_array<2, float>(line_data);
          return tf::distance2(pt, line);
        });

  m.def("distance_point_ray_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto ray = make_ray_from_array<2, float>(ray_data);
          return tf::distance(pt, ray);
        });

  m.def("distance2_point_ray_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto ray = make_ray_from_array<2, float>(ray_data);
          return tf::distance2(pt, ray);
        });

  m.def("distance_point_segment_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto seg = make_segment_from_array<2, float>(seg_data);
          return tf::distance(pt, seg);
        });

  m.def("distance2_point_segment_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto seg = make_segment_from_array<2, float>(seg_data);
          return tf::distance2(pt, seg);
        });

  m.def("distance_point_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 2>>
               poly_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return tf::distance(pt, poly);
        });

  m.def("distance2_point_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 2>>
               poly_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return tf::distance2(pt, poly);
        });

  m.def("distance_aabb_aabb_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               aabb0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               aabb1_data) {
          auto aabb0 = make_aabb_from_array<2, float>(aabb0_data);
          auto aabb1 = make_aabb_from_array<2, float>(aabb1_data);
          return tf::distance(aabb0, aabb1);
        });

  m.def("distance2_aabb_aabb_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               aabb0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               aabb1_data) {
          auto aabb0 = make_aabb_from_array<2, float>(aabb0_data);
          auto aabb1 = make_aabb_from_array<2, float>(aabb1_data);
          return tf::distance2(aabb0, aabb1);
        });

  m.def("distance_line_line_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line1_data) {
          auto line0 = make_line_from_array<2, float>(line0_data);
          auto line1 = make_line_from_array<2, float>(line1_data);
          return tf::distance(line0, line1);
        });

  m.def("distance2_line_line_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line1_data) {
          auto line0 = make_line_from_array<2, float>(line0_data);
          auto line1 = make_line_from_array<2, float>(line1_data);
          return tf::distance2(line0, line1);
        });

  m.def("distance_line_ray_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto line = make_line_from_array<2, float>(line_data);
          auto ray = make_ray_from_array<2, float>(ray_data);
          return tf::distance(line, ray);
        });

  m.def("distance2_line_ray_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto line = make_line_from_array<2, float>(line_data);
          auto ray = make_ray_from_array<2, float>(ray_data);
          return tf::distance2(line, ray);
        });

  m.def("distance_line_segment_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto line = make_line_from_array<2, float>(line_data);
          auto seg = make_segment_from_array<2, float>(seg_data);
          return tf::distance(line, seg);
        });

  m.def("distance2_line_segment_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto line = make_line_from_array<2, float>(line_data);
          auto seg = make_segment_from_array<2, float>(seg_data);
          return tf::distance2(line, seg);
        });

  m.def("distance_line_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 2>>
               poly_data) {
          auto line = make_line_from_array<2, float>(line_data);
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return tf::distance(line, poly);
        });

  m.def("distance2_line_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 2>>
               poly_data) {
          auto line = make_line_from_array<2, float>(line_data);
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return tf::distance2(line, poly);
        });

  m.def("distance_ray_ray_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray1_data) {
          auto ray0 = make_ray_from_array<2, float>(ray0_data);
          auto ray1 = make_ray_from_array<2, float>(ray1_data);
          return tf::distance(ray0, ray1);
        });

  m.def("distance2_ray_ray_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray1_data) {
          auto ray0 = make_ray_from_array<2, float>(ray0_data);
          auto ray1 = make_ray_from_array<2, float>(ray1_data);
          return tf::distance2(ray0, ray1);
        });

  m.def("distance_ray_segment_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          auto seg = make_segment_from_array<2, float>(seg_data);
          return tf::distance(ray, seg);
        });

  m.def("distance2_ray_segment_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          auto seg = make_segment_from_array<2, float>(seg_data);
          return tf::distance2(ray, seg);
        });

  m.def("distance_ray_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 2>>
               poly_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return tf::distance(ray, poly);
        });

  m.def("distance2_ray_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 2>>
               poly_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return tf::distance2(ray, poly);
        });

  m.def("distance_segment_segment_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg1_data) {
          auto seg0 = make_segment_from_array<2, float>(seg0_data);
          auto seg1 = make_segment_from_array<2, float>(seg1_data);
          return tf::distance(seg0, seg1);
        });

  m.def("distance2_segment_segment_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg1_data) {
          auto seg0 = make_segment_from_array<2, float>(seg0_data);
          auto seg1 = make_segment_from_array<2, float>(seg1_data);
          return tf::distance2(seg0, seg1);
        });

  m.def("distance_segment_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 2>>
               poly_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return tf::distance(seg, poly);
        });

  m.def("distance2_segment_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 2>>
               poly_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return tf::distance2(seg, poly);
        });

  m.def("distance_polygon_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 2>>
               poly0_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 2>>
               poly1_data) {
          auto poly0 = make_polygon_from_array<2, float>(poly0_data);
          auto poly1 = make_polygon_from_array<2, float>(poly1_data);
          return tf::distance(poly0, poly1);
        });

  m.def("distance2_polygon_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 2>>
               poly0_data,
           nanobind::ndarray<nanobind::numpy, const float,
                            nanobind::shape<-1, 2>>
               poly1_data) {
          auto poly0 = make_polygon_from_array<2, float>(poly0_data);
          auto poly1 = make_polygon_from_array<2, float>(poly1_data);
          return tf::distance2(poly0, poly1);
        });

}

} // namespace tf::py

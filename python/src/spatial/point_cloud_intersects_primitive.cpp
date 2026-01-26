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
#include <trueform/python/core/make_primitives.hpp>
#include <trueform/python/spatial/point_cloud.hpp>
#include <trueform/python/spatial/form_intersects_primitive.hpp>

namespace tf::py {

auto register_point_cloud_intersects_primitive(nanobind::module_ &m) -> void {

  // ============================================================================
  // PointCloud intersects primitives
  // Real types: float, double
  // Dims: 2D, 3D
  // Total: 2 × 2 = 4 point cloud types
  // Primitives: Point, Segment, Polygon, Ray, Line = 5 primitives
  // Total: 4 × 5 = 20 functions
  // ============================================================================

  // float, 2D
  m.def("intersects_point_cloud_point_float2d",
        [](point_cloud_wrapper<float, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          return form_intersects_primitive(cloud, pt);
        },
        nanobind::arg("point_cloud"), nanobind::arg("point"));

  m.def("intersects_point_cloud_segment_float2d",
        [](point_cloud_wrapper<float, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          return form_intersects_primitive(cloud, seg);
        },
        nanobind::arg("point_cloud"), nanobind::arg("segment"));

  m.def("intersects_point_cloud_polygon_float2d",
        [](point_cloud_wrapper<float, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return form_intersects_primitive(cloud, poly);
        },
        nanobind::arg("point_cloud"), nanobind::arg("polygon"));

  m.def("intersects_point_cloud_ray_float2d",
        [](point_cloud_wrapper<float, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          return form_intersects_primitive(cloud, ray);
        },
        nanobind::arg("point_cloud"), nanobind::arg("ray"));

  m.def("intersects_point_cloud_line_float2d",
        [](point_cloud_wrapper<float, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, float>(line_data);
          return form_intersects_primitive(cloud, line);
        },
        nanobind::arg("point_cloud"), nanobind::arg("line"));

  // float, 3D
  m.def("intersects_point_cloud_point_float3d",
        [](point_cloud_wrapper<float, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          return form_intersects_primitive(cloud, pt);
        },
        nanobind::arg("point_cloud"), nanobind::arg("point"));

  m.def("intersects_point_cloud_segment_float3d",
        [](point_cloud_wrapper<float, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          return form_intersects_primitive(cloud, seg);
        },
        nanobind::arg("point_cloud"), nanobind::arg("segment"));

  m.def("intersects_point_cloud_polygon_float3d",
        [](point_cloud_wrapper<float, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return form_intersects_primitive(cloud, poly);
        },
        nanobind::arg("point_cloud"), nanobind::arg("polygon"));

  m.def("intersects_point_cloud_ray_float3d",
        [](point_cloud_wrapper<float, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          return form_intersects_primitive(cloud, ray);
        },
        nanobind::arg("point_cloud"), nanobind::arg("ray"));

  m.def("intersects_point_cloud_line_float3d",
        [](point_cloud_wrapper<float, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, float>(line_data);
          return form_intersects_primitive(cloud, line);
        },
        nanobind::arg("point_cloud"), nanobind::arg("line"));

  // double, 2D
  m.def("intersects_point_cloud_point_double2d",
        [](point_cloud_wrapper<double, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          return form_intersects_primitive(cloud, pt);
        },
        nanobind::arg("point_cloud"), nanobind::arg("point"));

  m.def("intersects_point_cloud_segment_double2d",
        [](point_cloud_wrapper<double, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, double>(seg_data);
          return form_intersects_primitive(cloud, seg);
        },
        nanobind::arg("point_cloud"), nanobind::arg("segment"));

  m.def("intersects_point_cloud_polygon_double2d",
        [](point_cloud_wrapper<double, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return form_intersects_primitive(cloud, poly);
        },
        nanobind::arg("point_cloud"), nanobind::arg("polygon"));

  m.def("intersects_point_cloud_ray_double2d",
        [](point_cloud_wrapper<double, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, double>(ray_data);
          return form_intersects_primitive(cloud, ray);
        },
        nanobind::arg("point_cloud"), nanobind::arg("ray"));

  m.def("intersects_point_cloud_line_double2d",
        [](point_cloud_wrapper<double, 2> &cloud,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, double>(line_data);
          return form_intersects_primitive(cloud, line);
        },
        nanobind::arg("point_cloud"), nanobind::arg("line"));

  // double, 3D
  m.def("intersects_point_cloud_point_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          return form_intersects_primitive(cloud, pt);
        },
        nanobind::arg("point_cloud"), nanobind::arg("point"));

  m.def("intersects_point_cloud_segment_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, double>(seg_data);
          return form_intersects_primitive(cloud, seg);
        },
        nanobind::arg("point_cloud"), nanobind::arg("segment"));

  m.def("intersects_point_cloud_polygon_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return form_intersects_primitive(cloud, poly);
        },
        nanobind::arg("point_cloud"), nanobind::arg("polygon"));

  m.def("intersects_point_cloud_ray_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          return form_intersects_primitive(cloud, ray);
        },
        nanobind::arg("point_cloud"), nanobind::arg("ray"));

  m.def("intersects_point_cloud_line_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, double>(line_data);
          return form_intersects_primitive(cloud, line);
        },
        nanobind::arg("point_cloud"), nanobind::arg("line"));

  // ==== Plane (3D only) ====
  // float, 3D
  m.def("intersects_point_cloud_plane_float3d",
        [](point_cloud_wrapper<float, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, float>(plane_data);
          return form_intersects_primitive(cloud, plane);
        },
        nanobind::arg("point_cloud"), nanobind::arg("plane"));

  // double, 3D
  m.def("intersects_point_cloud_plane_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, double>(plane_data);
          return form_intersects_primitive(cloud, plane);
        },
        nanobind::arg("point_cloud"), nanobind::arg("plane"));
}

} // namespace tf::py

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
#include <trueform/python/spatial/edge_mesh.hpp>
#include <trueform/python/core/make_primitives.hpp>
#include <trueform/python/spatial/form_intersects_primitive.hpp>

namespace tf::py {

auto register_edge_mesh_intersects_primitive(nanobind::module_ &m) -> void {

  // ============================================================================
  // EdgeMesh intersects primitives
  // Index types: int, int64
  // Real types: float, double
  // Dims: 2D, 3D
  // Total: 2 × 2 × 2 = 8 edge mesh types
  // Primitives: Point, Segment, Polygon, Ray, Line = 5 primitives
  // Total: 8 × 5 = 40 functions
  // ============================================================================

  // int32, float, 2D
  m.def("intersects_edge_mesh_point_intfloat2d",
        [](edge_mesh_wrapper<int, float, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          return form_intersects_primitive(edge_mesh, pt);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("point"));

  m.def("intersects_edge_mesh_segment_intfloat2d",
        [](edge_mesh_wrapper<int, float, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          return form_intersects_primitive(edge_mesh, seg);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("segment"));

  m.def("intersects_edge_mesh_polygon_intfloat2d",
        [](edge_mesh_wrapper<int, float, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return form_intersects_primitive(edge_mesh, poly);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("polygon"));

  m.def("intersects_edge_mesh_ray_intfloat2d",
        [](edge_mesh_wrapper<int, float, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          return form_intersects_primitive(edge_mesh, ray);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("ray"));

  m.def("intersects_edge_mesh_line_intfloat2d",
        [](edge_mesh_wrapper<int, float, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, float>(line_data);
          return form_intersects_primitive(edge_mesh, line);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("line"));

  // int32, float, 3D
  m.def("intersects_edge_mesh_point_intfloat3d",
        [](edge_mesh_wrapper<int, float, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          return form_intersects_primitive(edge_mesh, pt);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("point"));

  m.def("intersects_edge_mesh_segment_intfloat3d",
        [](edge_mesh_wrapper<int, float, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          return form_intersects_primitive(edge_mesh, seg);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("segment"));

  m.def("intersects_edge_mesh_polygon_intfloat3d",
        [](edge_mesh_wrapper<int, float, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return form_intersects_primitive(edge_mesh, poly);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("polygon"));

  m.def("intersects_edge_mesh_ray_intfloat3d",
        [](edge_mesh_wrapper<int, float, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          return form_intersects_primitive(edge_mesh, ray);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("ray"));

  m.def("intersects_edge_mesh_line_intfloat3d",
        [](edge_mesh_wrapper<int, float, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, float>(line_data);
          return form_intersects_primitive(edge_mesh, line);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("line"));

  // int32, double, 2D
  m.def("intersects_edge_mesh_point_intdouble2d",
        [](edge_mesh_wrapper<int, double, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          return form_intersects_primitive(edge_mesh, pt);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("point"));

  m.def("intersects_edge_mesh_segment_intdouble2d",
        [](edge_mesh_wrapper<int, double, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, double>(seg_data);
          return form_intersects_primitive(edge_mesh, seg);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("segment"));

  m.def("intersects_edge_mesh_polygon_intdouble2d",
        [](edge_mesh_wrapper<int, double, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return form_intersects_primitive(edge_mesh, poly);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("polygon"));

  m.def("intersects_edge_mesh_ray_intdouble2d",
        [](edge_mesh_wrapper<int, double, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, double>(ray_data);
          return form_intersects_primitive(edge_mesh, ray);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("ray"));

  m.def("intersects_edge_mesh_line_intdouble2d",
        [](edge_mesh_wrapper<int, double, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, double>(line_data);
          return form_intersects_primitive(edge_mesh, line);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("line"));

  // int32, double, 3D
  m.def("intersects_edge_mesh_point_intdouble3d",
        [](edge_mesh_wrapper<int, double, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          return form_intersects_primitive(edge_mesh, pt);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("point"));

  m.def("intersects_edge_mesh_segment_intdouble3d",
        [](edge_mesh_wrapper<int, double, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, double>(seg_data);
          return form_intersects_primitive(edge_mesh, seg);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("segment"));

  m.def("intersects_edge_mesh_polygon_intdouble3d",
        [](edge_mesh_wrapper<int, double, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return form_intersects_primitive(edge_mesh, poly);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("polygon"));

  m.def("intersects_edge_mesh_ray_intdouble3d",
        [](edge_mesh_wrapper<int, double, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          return form_intersects_primitive(edge_mesh, ray);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("ray"));

  m.def("intersects_edge_mesh_line_intdouble3d",
        [](edge_mesh_wrapper<int, double, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, double>(line_data);
          return form_intersects_primitive(edge_mesh, line);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("line"));

  // int64, float, 2D
  m.def("intersects_edge_mesh_point_int64float2d",
        [](edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          return form_intersects_primitive(edge_mesh, pt);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("point"));

  m.def("intersects_edge_mesh_segment_int64float2d",
        [](edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          return form_intersects_primitive(edge_mesh, seg);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("segment"));

  m.def("intersects_edge_mesh_polygon_int64float2d",
        [](edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return form_intersects_primitive(edge_mesh, poly);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("polygon"));

  m.def("intersects_edge_mesh_ray_int64float2d",
        [](edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          return form_intersects_primitive(edge_mesh, ray);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("ray"));

  m.def("intersects_edge_mesh_line_int64float2d",
        [](edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, float>(line_data);
          return form_intersects_primitive(edge_mesh, line);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("line"));

  // int64, float, 3D
  m.def("intersects_edge_mesh_point_int64float3d",
        [](edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          return form_intersects_primitive(edge_mesh, pt);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("point"));

  m.def("intersects_edge_mesh_segment_int64float3d",
        [](edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          return form_intersects_primitive(edge_mesh, seg);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("segment"));

  m.def("intersects_edge_mesh_polygon_int64float3d",
        [](edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return form_intersects_primitive(edge_mesh, poly);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("polygon"));

  m.def("intersects_edge_mesh_ray_int64float3d",
        [](edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          return form_intersects_primitive(edge_mesh, ray);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("ray"));

  m.def("intersects_edge_mesh_line_int64float3d",
        [](edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, float>(line_data);
          return form_intersects_primitive(edge_mesh, line);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("line"));

  // int64, double, 2D
  m.def("intersects_edge_mesh_point_int64double2d",
        [](edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          return form_intersects_primitive(edge_mesh, pt);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("point"));

  m.def("intersects_edge_mesh_segment_int64double2d",
        [](edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, double>(seg_data);
          return form_intersects_primitive(edge_mesh, seg);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("segment"));

  m.def("intersects_edge_mesh_polygon_int64double2d",
        [](edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return form_intersects_primitive(edge_mesh, poly);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("polygon"));

  m.def("intersects_edge_mesh_ray_int64double2d",
        [](edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, double>(ray_data);
          return form_intersects_primitive(edge_mesh, ray);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("ray"));

  m.def("intersects_edge_mesh_line_int64double2d",
        [](edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, double>(line_data);
          return form_intersects_primitive(edge_mesh, line);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("line"));

  // int64, double, 3D
  m.def("intersects_edge_mesh_point_int64double3d",
        [](edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          return form_intersects_primitive(edge_mesh, pt);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("point"));

  m.def("intersects_edge_mesh_segment_int64double3d",
        [](edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, double>(seg_data);
          return form_intersects_primitive(edge_mesh, seg);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("segment"));

  m.def("intersects_edge_mesh_polygon_int64double3d",
        [](edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return form_intersects_primitive(edge_mesh, poly);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("polygon"));

  m.def("intersects_edge_mesh_ray_int64double3d",
        [](edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          return form_intersects_primitive(edge_mesh, ray);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("ray"));

  m.def("intersects_edge_mesh_line_int64double3d",
        [](edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, double>(line_data);
          return form_intersects_primitive(edge_mesh, line);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("line"));

  // ==== Plane (3D only) ====
  // int32, float, 3D
  m.def("intersects_edge_mesh_plane_intfloat3d",
        [](edge_mesh_wrapper<int, float, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, float>(plane_data);
          return form_intersects_primitive(edge_mesh, plane);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("plane"));

  // int32, double, 3D
  m.def("intersects_edge_mesh_plane_intdouble3d",
        [](edge_mesh_wrapper<int, double, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, double>(plane_data);
          return form_intersects_primitive(edge_mesh, plane);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("plane"));

  // int64, float, 3D
  m.def("intersects_edge_mesh_plane_int64float3d",
        [](edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, float>(plane_data);
          return form_intersects_primitive(edge_mesh, plane);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("plane"));

  // int64, double, 3D
  m.def("intersects_edge_mesh_plane_int64double3d",
        [](edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, double>(plane_data);
          return form_intersects_primitive(edge_mesh, plane);
        },
        nanobind::arg("edge_mesh"), nanobind::arg("plane"));
}

} // namespace tf::py

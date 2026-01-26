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
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/form_intersects_primitive.hpp>

namespace tf::py {

auto register_mesh_intersects_primitive(nanobind::module_ &m) -> void {

  // ============================================================================
  // Mesh intersects primitives
  // Index types: int, int64
  // Real types: float, double
  // Ngon: 3 (triangles), dynamic
  // Dims: 2D, 3D
  // Total: 2 × 2 × 2 × 2 = 16 mesh types
  // Primitives: Point, Segment, Polygon, Ray, Line = 5 primitives
  // Total: 16 × 5 = 80 functions
  // ============================================================================

  // int32, float, triangle, 2D
  m.def("intersects_mesh_point_int3float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int3float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int3float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int3float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int3float2d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int32, float, triangle, 3D
  m.def("intersects_mesh_point_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int32, float, dynamic, 2D
  m.def("intersects_mesh_point_intdynfloat2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_intdynfloat2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_intdynfloat2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_intdynfloat2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_intdynfloat2d",
        [](mesh_wrapper<int, float, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int32, float, dynamic, 3D
  m.def("intersects_mesh_point_intdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_intdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_intdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_intdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_intdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int32, double, triangle, 2D
  m.def("intersects_mesh_point_int3double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int3double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int3double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int3double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int3double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int32, double, triangle, 3D
  m.def("intersects_mesh_point_int3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int32, double, dynamic, 2D
  m.def("intersects_mesh_point_intdyndouble2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_intdyndouble2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_intdyndouble2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_intdyndouble2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_intdyndouble2d",
        [](mesh_wrapper<int, double, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int32, double, dynamic, 3D
  m.def("intersects_mesh_point_intdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_intdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_intdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_intdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_intdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, float, triangle, 2D
  m.def("intersects_mesh_point_int643float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int643float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int643float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int643float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int643float2d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, float, triangle, 3D
  m.def("intersects_mesh_point_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, float, dynamic, 2D
  m.def("intersects_mesh_point_int64dynfloat2d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int64dynfloat2d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int64dynfloat2d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int64dynfloat2d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int64dynfloat2d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, float, dynamic, 3D
  m.def("intersects_mesh_point_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, double, triangle, 2D
  m.def("intersects_mesh_point_int643double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int643double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int643double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int643double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int643double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, double, triangle, 3D
  m.def("intersects_mesh_point_int643double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int643double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int643double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int643double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int643double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, double, dynamic, 2D
  m.def("intersects_mesh_point_int64dyndouble2d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int64dyndouble2d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int64dyndouble2d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int64dyndouble2d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int64dyndouble2d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, double, dynamic, 3D
  m.def("intersects_mesh_point_int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // ==== Plane (3D only) ====
  // int32, float, triangle, 3D
  m.def("intersects_mesh_plane_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, float>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));

  // int32, float, dynamic, 3D
  m.def("intersects_mesh_plane_intdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, float>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));

  // int32, double, triangle, 3D
  m.def("intersects_mesh_plane_int3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, double>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));

  // int32, double, dynamic, 3D
  m.def("intersects_mesh_plane_intdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, double>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));

  // int64, float, triangle, 3D
  m.def("intersects_mesh_plane_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, float>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));

  // int64, float, dynamic, 3D
  m.def("intersects_mesh_plane_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, float>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));

  // int64, double, triangle, 3D
  m.def("intersects_mesh_plane_int643double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, double>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));

  // int64, double, dynamic, 3D
  m.def("intersects_mesh_plane_int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, double>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));
}

} // namespace tf::py

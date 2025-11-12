/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <trueform/python/core/make_primitives.hpp>
#include <trueform/python/core/mesh.hpp>
#include <trueform/python/spatial/form_intersects_primitive.hpp>

namespace tf::py {

auto register_mesh_intersects_primitive(nanobind::module_ &m) -> void {

  // ============================================================================
  // Mesh intersects primitives
  // Index types: int, int64
  // Real types: float, double
  // Ngon: 3 (triangles), 4 (quads)
  // Dims: 2D, 3D
  // Total: 2 × 2 × 2 × 2 = 16 mesh types
  // Primitives: Point, Segment, Polygon, Ray, Line = 5 primitives
  // Total: 16 × 5 = 80 functions
  // ============================================================================

  // int32, float, triangle, 2D
  m.def("intersects_mesh_point_intfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_intfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_intfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_intfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_intfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int32, float, triangle, 3D
  m.def("intersects_mesh_point_intfloat33d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_intfloat33d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_intfloat33d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_intfloat33d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_intfloat33d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int32, float, quad, 2D
  m.def("intersects_mesh_point_intfloat42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_intfloat42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_intfloat42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_intfloat42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_intfloat42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int32, float, quad, 3D
  m.def("intersects_mesh_point_intfloat43d",
        [](mesh_wrapper<int, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_intfloat43d",
        [](mesh_wrapper<int, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_intfloat43d",
        [](mesh_wrapper<int, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_intfloat43d",
        [](mesh_wrapper<int, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_intfloat43d",
        [](mesh_wrapper<int, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int32, double, triangle, 2D
  m.def("intersects_mesh_point_intdouble32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_intdouble32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_intdouble32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_intdouble32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_intdouble32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int32, double, triangle, 3D
  m.def("intersects_mesh_point_intdouble33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_intdouble33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_intdouble33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_intdouble33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_intdouble33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int32, double, quad, 2D
  m.def("intersects_mesh_point_intdouble42d",
        [](mesh_wrapper<int, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_intdouble42d",
        [](mesh_wrapper<int, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_intdouble42d",
        [](mesh_wrapper<int, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_intdouble42d",
        [](mesh_wrapper<int, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_intdouble42d",
        [](mesh_wrapper<int, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int32, double, quad, 3D
  m.def("intersects_mesh_point_intdouble43d",
        [](mesh_wrapper<int, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_intdouble43d",
        [](mesh_wrapper<int, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_intdouble43d",
        [](mesh_wrapper<int, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_intdouble43d",
        [](mesh_wrapper<int, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_intdouble43d",
        [](mesh_wrapper<int, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, float, triangle, 2D
  m.def("intersects_mesh_point_int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, float, triangle, 3D
  m.def("intersects_mesh_point_int64float33d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int64float33d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int64float33d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int64float33d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int64float33d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, float, quad, 2D
  m.def("intersects_mesh_point_int64float42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int64float42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int64float42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int64float42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int64float42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, float, quad, 3D
  m.def("intersects_mesh_point_int64float43d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int64float43d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, float>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int64float43d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int64float43d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, float>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int64float43d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, float>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, double, triangle, 2D
  m.def("intersects_mesh_point_int64double32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int64double32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int64double32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int64double32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int64double32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, double, triangle, 3D
  m.def("intersects_mesh_point_int64double33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int64double33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int64double33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int64double33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int64double33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, double, quad, 2D
  m.def("intersects_mesh_point_int64double42d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int64double42d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               seg_data) {
          auto seg = make_segment_from_array<2, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int64double42d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int64double42d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               ray_data) {
          auto ray = make_ray_from_array<2, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int64double42d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 2>>
               line_data) {
          auto line = make_line_from_array<2, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // int64, double, quad, 3D
  m.def("intersects_mesh_point_int64double43d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          return form_intersects_primitive(mesh, pt);
        },
        nanobind::arg("mesh"), nanobind::arg("point"));

  m.def("intersects_mesh_segment_int64double43d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               seg_data) {
          auto seg = make_segment_from_array<3, double>(seg_data);
          return form_intersects_primitive(mesh, seg);
        },
        nanobind::arg("mesh"), nanobind::arg("segment"));

  m.def("intersects_mesh_polygon_int64double43d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return form_intersects_primitive(mesh, poly);
        },
        nanobind::arg("mesh"), nanobind::arg("polygon"));

  m.def("intersects_mesh_ray_int64double43d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               ray_data) {
          auto ray = make_ray_from_array<3, double>(ray_data);
          return form_intersects_primitive(mesh, ray);
        },
        nanobind::arg("mesh"), nanobind::arg("ray"));

  m.def("intersects_mesh_line_int64double43d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<2, 3>>
               line_data) {
          auto line = make_line_from_array<3, double>(line_data);
          return form_intersects_primitive(mesh, line);
        },
        nanobind::arg("mesh"), nanobind::arg("line"));

  // ==== Plane (3D only) ====
  // int32, float, triangle, 3D
  m.def("intersects_mesh_plane_intfloat33d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, float>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));

  // int32, float, quad, 3D
  m.def("intersects_mesh_plane_intfloat43d",
        [](mesh_wrapper<int, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, float>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));

  // int32, double, triangle, 3D
  m.def("intersects_mesh_plane_intdouble33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, double>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));

  // int32, double, quad, 3D
  m.def("intersects_mesh_plane_intdouble43d",
        [](mesh_wrapper<int, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, double>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));

  // int64, float, triangle, 3D
  m.def("intersects_mesh_plane_int64float33d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, float>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));

  // int64, float, quad, 3D
  m.def("intersects_mesh_plane_int64float43d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, float>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));

  // int64, double, triangle, 3D
  m.def("intersects_mesh_plane_int64double33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, double>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));

  // int64, double, quad, 3D
  m.def("intersects_mesh_plane_int64double43d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto plane = make_plane_from_array<3, double>(plane_data);
          return form_intersects_primitive(mesh, plane);
        },
        nanobind::arg("mesh"), nanobind::arg("plane"));
}

} // namespace tf::py

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <trueform/python/core/make_primitives.hpp>
#include <trueform/python/core/mesh.hpp>
#include <trueform/python/spatial/neighbor_search.hpp>

namespace tf::py {

auto register_mesh_neighbor_search(nanobind::module_ &m) -> void {

  // ============================================================================
  // Non-KNN neighbor search (single nearest neighbor)
  // MeshWrapperIntFloat32D (int, float, tri, 2D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_intfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               query,
           std::optional<float> radius) {
          return neighbor_search<float, 2>(
              mesh, make_point_from_array<2, float>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_intfloat32d",
      [](mesh_wrapper<int, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_segment_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_mesh_polygon_intfloat32d",
      [](mesh_wrapper<int, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_polygon_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_intfloat32d",
      [](mesh_wrapper<int, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_ray_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_intfloat32d",
      [](mesh_wrapper<int, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_line_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperIntFloat33D (int, float, tri, 3D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_intfloat33d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               query,
           std::optional<float> radius) {
          return neighbor_search<float, 3>(
              mesh, make_point_from_array<3, float>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_intfloat33d",
      [](mesh_wrapper<int, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_segment_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_mesh_polygon_intfloat33d",
      [](mesh_wrapper<int, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_polygon_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_intfloat33d",
      [](mesh_wrapper<int, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_ray_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_intfloat33d",
      [](mesh_wrapper<int, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_line_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperIntFloat42D (int, float, quad, 2D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_intfloat42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               query,
           std::optional<float> radius) {
          return neighbor_search<float, 2>(
              mesh, make_point_from_array<2, float>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_intfloat42d",
      [](mesh_wrapper<int, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_segment_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_mesh_polygon_intfloat42d",
      [](mesh_wrapper<int, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_polygon_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_intfloat42d",
      [](mesh_wrapper<int, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_ray_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_intfloat42d",
      [](mesh_wrapper<int, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_line_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperIntFloat43D (int, float, quad, 3D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_intfloat43d",
        [](mesh_wrapper<int, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               query,
           std::optional<float> radius) {
          return neighbor_search<float, 3>(
              mesh, make_point_from_array<3, float>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_intfloat43d",
      [](mesh_wrapper<int, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_segment_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_mesh_polygon_intfloat43d",
      [](mesh_wrapper<int, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_polygon_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_intfloat43d",
      [](mesh_wrapper<int, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_ray_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_intfloat43d",
      [](mesh_wrapper<int, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_line_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperIntDouble32D (int, double, tri, 2D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_intdouble32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 2>(
              mesh, make_point_from_array<2, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_intdouble32d",
      [](mesh_wrapper<int, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_segment_from_array<2, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def("neighbor_search_mesh_polygon_intdouble32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<-1, 2>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 2>(
              mesh, make_polygon_from_array<2, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_intdouble32d",
      [](mesh_wrapper<int, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_ray_from_array<2, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_intdouble32d",
      [](mesh_wrapper<int, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_line_from_array<2, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperIntDouble33D (int, double, tri, 3D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_intdouble33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 3>(
              mesh, make_point_from_array<3, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_intdouble33d",
      [](mesh_wrapper<int, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_segment_from_array<3, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def("neighbor_search_mesh_polygon_intdouble33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<-1, 3>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 3>(
              mesh, make_polygon_from_array<3, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_intdouble33d",
      [](mesh_wrapper<int, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_ray_from_array<3, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_intdouble33d",
      [](mesh_wrapper<int, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_line_from_array<3, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperIntDouble42D (int, double, quad, 2D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_intdouble42d",
        [](mesh_wrapper<int, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 2>(
              mesh, make_point_from_array<2, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_intdouble42d",
      [](mesh_wrapper<int, double, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_segment_from_array<2, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def("neighbor_search_mesh_polygon_intdouble42d",
        [](mesh_wrapper<int, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<-1, 2>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 2>(
              mesh, make_polygon_from_array<2, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_intdouble42d",
      [](mesh_wrapper<int, double, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_ray_from_array<2, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_intdouble42d",
      [](mesh_wrapper<int, double, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_line_from_array<2, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperIntDouble43D (int, double, quad, 3D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_intdouble43d",
        [](mesh_wrapper<int, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 3>(
              mesh, make_point_from_array<3, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_intdouble43d",
      [](mesh_wrapper<int, double, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_segment_from_array<3, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def("neighbor_search_mesh_polygon_intdouble43d",
        [](mesh_wrapper<int, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<-1, 3>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 3>(
              mesh, make_polygon_from_array<3, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_intdouble43d",
      [](mesh_wrapper<int, double, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_ray_from_array<3, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_intdouble43d",
      [](mesh_wrapper<int, double, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_line_from_array<3, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperInt64Float32D (int64, float, tri, 2D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               query,
           std::optional<float> radius) {
          return neighbor_search<float, 2>(
              mesh, make_point_from_array<2, float>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_int64float32d",
      [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_segment_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_mesh_polygon_int64float32d",
      [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_polygon_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_int64float32d",
      [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_ray_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_int64float32d",
      [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_line_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperInt64Float33D (int64, float, tri, 3D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_int64float33d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               query,
           std::optional<float> radius) {
          return neighbor_search<float, 3>(
              mesh, make_point_from_array<3, float>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_int64float33d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_segment_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_mesh_polygon_int64float33d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_polygon_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_int64float33d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_ray_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_int64float33d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_line_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperInt64Float42D (int64, float, quad, 2D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_int64float42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               query,
           std::optional<float> radius) {
          return neighbor_search<float, 2>(
              mesh, make_point_from_array<2, float>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_int64float42d",
      [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_segment_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_mesh_polygon_int64float42d",
      [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_polygon_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_int64float42d",
      [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_ray_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_int64float42d",
      [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_line_from_array<2, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperInt64Float43D (int64, float, quad, 3D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_int64float43d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               query,
           std::optional<float> radius) {
          return neighbor_search<float, 3>(
              mesh, make_point_from_array<3, float>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_int64float43d",
      [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_segment_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def(
      "neighbor_search_mesh_polygon_int64float43d",
      [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_polygon_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_int64float43d",
      [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_ray_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_int64float43d",
      [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_line_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperInt64Double32D (int64, double, tri, 2D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_int64double32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 2>(
              mesh, make_point_from_array<2, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_int64double32d",
      [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_segment_from_array<2, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def("neighbor_search_mesh_polygon_int64double32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<-1, 2>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 2>(
              mesh, make_polygon_from_array<2, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_int64double32d",
      [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_ray_from_array<2, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_int64double32d",
      [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_line_from_array<2, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperInt64Double33D (int64, double, tri, 3D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_int64double33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 3>(
              mesh, make_point_from_array<3, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_int64double33d",
      [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_segment_from_array<3, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def("neighbor_search_mesh_polygon_int64double33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<-1, 3>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 3>(
              mesh, make_polygon_from_array<3, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_int64double33d",
      [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_ray_from_array<3, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_int64double33d",
      [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_line_from_array<3, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperInt64Double42D (int64, double, quad, 2D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_int64double42d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 2>(
              mesh, make_point_from_array<2, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_int64double42d",
      [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_segment_from_array<2, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def("neighbor_search_mesh_polygon_int64double42d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<-1, 2>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 2>(
            mesh, make_polygon_from_array<2, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_int64double42d",
      [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_ray_from_array<2, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_int64double42d",
      [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_line_from_array<2, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // MeshWrapperInt64Double43D (int64, double, quad, 3D)
  // ============================================================================

  // Point queries
  m.def("neighbor_search_mesh_point_int64double43d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 3>(
              mesh, make_point_from_array<3, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Segment queries
  m.def(
      "neighbor_search_mesh_segment_int64double43d",
      [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_segment_from_array<3, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Polygon queries
  m.def("neighbor_search_mesh_polygon_int64double43d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double,
                             nanobind::shape<-1, 3>>
               query,
           std::optional<double> radius) {
          return neighbor_search<double, 3>(
              mesh, make_polygon_from_array<3, double>(query), radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("radius").none() = nanobind::none());

  // Ray queries
  m.def(
      "neighbor_search_mesh_ray_int64double43d",
      [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_ray_from_array<3, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Line queries
  m.def(
      "neighbor_search_mesh_line_int64double43d",
      [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_line_from_array<3, double>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // ============================================================================
  // KNN neighbor search (k nearest neighbors)
  // All 16 wrapper types with all 5 query types
  // ============================================================================

  // KNN: MeshWrapperIntFloat32D (int, float, tri, 2D)
  m.def("neighbor_search_mesh_knn_point_intfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               query,
           int k, std::optional<float> radius) {
          return neighbor_search<float, 2>(
              mesh, make_point_from_array<2, float>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_intfloat32d",
      [](mesh_wrapper<int, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_segment_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_intfloat32d",
      [](mesh_wrapper<int, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_polygon_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_intfloat32d",
      [](mesh_wrapper<int, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_ray_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_intfloat32d",
      [](mesh_wrapper<int, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_line_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperIntFloat33D (int, float, tri, 3D)
  m.def("neighbor_search_mesh_knn_point_intfloat33d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               query,
           int k, std::optional<float> radius) {
          return neighbor_search<float, 3>(
              mesh, make_point_from_array<3, float>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_intfloat33d",
      [](mesh_wrapper<int, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_segment_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_intfloat33d",
      [](mesh_wrapper<int, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_polygon_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_intfloat33d",
      [](mesh_wrapper<int, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_ray_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_intfloat33d",
      [](mesh_wrapper<int, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_line_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperIntFloat42D (int, float, quad, 2D)
  m.def("neighbor_search_mesh_knn_point_intfloat42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               query,
           int k, std::optional<float> radius) {
          return neighbor_search<float, 2>(
              mesh, make_point_from_array<2, float>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_intfloat42d",
      [](mesh_wrapper<int, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_segment_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_intfloat42d",
      [](mesh_wrapper<int, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_polygon_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_intfloat42d",
      [](mesh_wrapper<int, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_ray_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_intfloat42d",
      [](mesh_wrapper<int, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_line_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperIntFloat43D (int, float, quad, 3D)
  m.def("neighbor_search_mesh_knn_point_intfloat43d",
        [](mesh_wrapper<int, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               query,
           int k, std::optional<float> radius) {
          return neighbor_search<float, 3>(
              mesh, make_point_from_array<3, float>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_intfloat43d",
      [](mesh_wrapper<int, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_segment_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_intfloat43d",
      [](mesh_wrapper<int, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_polygon_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_intfloat43d",
      [](mesh_wrapper<int, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_ray_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_intfloat43d",
      [](mesh_wrapper<int, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_line_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperIntDouble32D (int, double, tri, 2D)
  m.def("neighbor_search_mesh_knn_point_intdouble32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               query,
           int k, std::optional<double> radius) {
          return neighbor_search<double, 2>(
              mesh, make_point_from_array<2, double>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_intdouble32d",
      [](mesh_wrapper<int, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_segment_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_intdouble32d",
      [](mesh_wrapper<int, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double,
                           nanobind::shape<-1, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_polygon_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_intdouble32d",
      [](mesh_wrapper<int, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_ray_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_intdouble32d",
      [](mesh_wrapper<int, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_line_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperIntDouble33D (int, double, tri, 3D)
  m.def("neighbor_search_mesh_knn_point_intdouble33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               query,
           int k, std::optional<double> radius) {
          return neighbor_search<double, 3>(
              mesh, make_point_from_array<3, double>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_intdouble33d",
      [](mesh_wrapper<int, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_segment_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_intdouble33d",
      [](mesh_wrapper<int, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double,
                           nanobind::shape<-1, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_polygon_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_intdouble33d",
      [](mesh_wrapper<int, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_ray_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_intdouble33d",
      [](mesh_wrapper<int, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_line_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperIntDouble42D (int, double, quad, 2D)
  m.def("neighbor_search_mesh_knn_point_intdouble42d",
        [](mesh_wrapper<int, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               query,
           int k, std::optional<double> radius) {
          return neighbor_search<double, 2>(
              mesh, make_point_from_array<2, double>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_intdouble42d",
      [](mesh_wrapper<int, double, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_segment_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_intdouble42d",
      [](mesh_wrapper<int, double, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double,
                           nanobind::shape<-1, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_polygon_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_intdouble42d",
      [](mesh_wrapper<int, double, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_ray_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_intdouble42d",
      [](mesh_wrapper<int, double, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_line_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperIntDouble43D (int, double, quad, 3D)
  m.def("neighbor_search_mesh_knn_point_intdouble43d",
        [](mesh_wrapper<int, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               query,
           int k, std::optional<double> radius) {
          return neighbor_search<double, 3>(
              mesh, make_point_from_array<3, double>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_intdouble43d",
      [](mesh_wrapper<int, double, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_segment_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_intdouble43d",
      [](mesh_wrapper<int, double, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double,
                           nanobind::shape<-1, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_polygon_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_intdouble43d",
      [](mesh_wrapper<int, double, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_ray_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_intdouble43d",
      [](mesh_wrapper<int, double, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_line_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperInt64Float32D (int64, float, tri, 2D)
  m.def("neighbor_search_mesh_knn_point_int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               query,
           int k, std::optional<float> radius) {
          return neighbor_search<float, 2>(
              mesh, make_point_from_array<2, float>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_int64float32d",
      [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_segment_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_int64float32d",
      [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_polygon_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_int64float32d",
      [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_ray_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_int64float32d",
      [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_line_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperInt64Float33D (int64, float, tri, 3D)
  m.def("neighbor_search_mesh_knn_point_int64float33d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               query,
           int k, std::optional<float> radius) {
          return neighbor_search<float, 3>(
              mesh, make_point_from_array<3, float>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_int64float33d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_segment_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_int64float33d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_polygon_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_int64float33d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_ray_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_int64float33d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_line_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperInt64Float42D (int64, float, quad, 2D)
  m.def("neighbor_search_mesh_knn_point_int64float42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               query,
           int k, std::optional<float> radius) {
          return neighbor_search<float, 2>(
              mesh, make_point_from_array<2, float>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_int64float42d",
      [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_segment_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_int64float42d",
      [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_polygon_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_int64float42d",
      [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_ray_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_int64float42d",
      [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 2>(
            mesh, make_line_from_array<2, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperInt64Float43D (int64, float, quad, 3D)
  m.def("neighbor_search_mesh_knn_point_int64float43d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               query,
           int k, std::optional<float> radius) {
          return neighbor_search<float, 3>(
              mesh, make_point_from_array<3, float>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_int64float43d",
      [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_segment_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_int64float43d",
      [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_polygon_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_int64float43d",
      [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_ray_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_int64float43d",
      [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_line_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperInt64Double32D (int64, double, tri, 2D)
  m.def("neighbor_search_mesh_knn_point_int64double32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               query,
           int k, std::optional<double> radius) {
          return neighbor_search<double, 2>(
              mesh, make_point_from_array<2, double>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_int64double32d",
      [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_segment_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_int64double32d",
      [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double,
                           nanobind::shape<-1, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_polygon_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_int64double32d",
      [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_ray_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_int64double32d",
      [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_line_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperInt64Double33D (int64, double, tri, 3D)
  m.def("neighbor_search_mesh_knn_point_int64double33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               query,
           int k, std::optional<double> radius) {
          return neighbor_search<double, 3>(
              mesh, make_point_from_array<3, double>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_int64double33d",
      [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_segment_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_int64double33d",
      [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double,
                           nanobind::shape<-1, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_polygon_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_int64double33d",
      [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_ray_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_int64double33d",
      [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_line_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperInt64Double42D (int64, double, quad, 2D)
  m.def("neighbor_search_mesh_knn_point_int64double42d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               query,
           int k, std::optional<double> radius) {
          return neighbor_search<double, 2>(
              mesh, make_point_from_array<2, double>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_int64double42d",
      [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_segment_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_int64double42d",
      [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double,
                           nanobind::shape<-1, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_polygon_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_int64double42d",
      [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_ray_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_int64double42d",
      [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 2>(
            mesh, make_line_from_array<2, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  // KNN: MeshWrapperInt64Double43D (int64, double, quad, 3D)
  m.def("neighbor_search_mesh_knn_point_int64double43d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               query,
           int k, std::optional<double> radius) {
          return neighbor_search<double, 3>(
              mesh, make_point_from_array<3, double>(query), k, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("query"),
        nanobind::arg("k"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_segment_int64double43d",
      [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_segment_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_polygon_int64double43d",
      [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double,
                           nanobind::shape<-1, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_polygon_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_ray_int64double43d",
      [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_ray_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def(
      "neighbor_search_mesh_knn_line_int64double43d",
      [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             query,
         int k, std::optional<double> radius) {
        return neighbor_search<double, 3>(
            mesh, make_line_from_array<3, double>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

}

} // namespace tf::py

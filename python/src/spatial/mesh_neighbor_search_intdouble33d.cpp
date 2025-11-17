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
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/neighbor_search.hpp>

namespace tf::py {

auto register_mesh_neighbor_search_intdouble33d(nanobind::module_ &m) -> void {

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

}

} // namespace tf::py

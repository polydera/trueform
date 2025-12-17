/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
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

auto register_mesh_neighbor_search_intfloatdyn3d(nanobind::module_ &m) -> void {

  // Point queries
  m.def("neighbor_search_mesh_point_intfloatdyn3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
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
      "neighbor_search_mesh_segment_intfloatdyn3d",
      [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
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
      "neighbor_search_mesh_polygon_intfloatdyn3d",
      [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
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
      "neighbor_search_mesh_ray_intfloatdyn3d",
      [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
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
      "neighbor_search_mesh_line_intfloatdyn3d",
      [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_line_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  // Plane queries
  m.def(
      "neighbor_search_mesh_plane_intfloatdyn3d",
      [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
             query,
         std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_plane_from_array<3, float>(query), radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_knn_point_intfloatdyn3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
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
      "neighbor_search_mesh_knn_segment_intfloatdyn3d",
      [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
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
      "neighbor_search_mesh_knn_polygon_intfloatdyn3d",
      [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
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
      "neighbor_search_mesh_knn_ray_intfloatdyn3d",
      [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
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
      "neighbor_search_mesh_knn_line_intfloatdyn3d",
      [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
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

  m.def(
      "neighbor_search_mesh_knn_plane_intfloatdyn3d",
      [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
             query,
         int k, std::optional<float> radius) {
        return neighbor_search<float, 3>(
            mesh, make_plane_from_array<3, float>(query), k, radius);
      },
      nanobind::arg("mesh"),
      nanobind::arg("query"),
      nanobind::arg("k"),
      nanobind::arg("radius").none() = nanobind::none());

}

} // namespace tf::py

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
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <trueform/core/aabb_from.hpp>
#include <trueform/core/distance.hpp>
#include <trueform/core/intersects.hpp>
#include <trueform/core/sphere.hpp>
#include <trueform/python/core/make_primitives.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/gather_ids.hpp>

namespace tf::py {

auto register_mesh_gather_ids_primitive_float3d(nanobind::module_ &m) -> void {

  namespace nb = nanobind;

  // ============================================================================
  // gather_ids - int, float, ngon=3, 3D
  // ============================================================================

  // Point - int3float3d
  m.def("gather_ids_point_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_point_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Segment - int3float3d
  m.def("gather_ids_segment_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_segment_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Polygon - int3float3d
  m.def("gather_ids_polygon_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<-1, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_polygon_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Ray - int3float3d
  m.def("gather_ids_ray_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_ray_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto aabb_pred = [&query](const auto &aabb) {
              return tf::intersects(query, aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto aabb_pred = [&query, threshold2](const auto &aabb) {
              return tf::distance2(
                  tf::make_sphere(aabb.center(), aabb.diagonal().length() / 2), query) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Line - int3float3d
  m.def("gather_ids_line_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_line_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto aabb_pred = [&query](const auto &aabb) {
              return tf::intersects(query, aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto aabb_pred = [&query, threshold2](const auto &aabb) {
              return tf::distance2(
                  tf::make_sphere(aabb.center(), aabb.diagonal().length() / 2), query) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // ============================================================================
  // gather_ids - int, float, dynamic, 3D
  // ============================================================================

  // Point - intdynfloat3d
  m.def("gather_ids_point_intdynfloat3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_point_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Segment - intdynfloat3d
  m.def("gather_ids_segment_intdynfloat3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_segment_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Polygon - intdynfloat3d
  m.def("gather_ids_polygon_intdynfloat3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<-1, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_polygon_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Ray - intdynfloat3d
  m.def("gather_ids_ray_intdynfloat3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_ray_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto aabb_pred = [&query](const auto &aabb) {
              return tf::intersects(query, aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto aabb_pred = [&query, threshold2](const auto &aabb) {
              return tf::distance2(
                  tf::make_sphere(aabb.center(), aabb.diagonal().length() / 2), query) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Line - intdynfloat3d
  m.def("gather_ids_line_intdynfloat3d",
        [](mesh_wrapper<int, float, tf::dynamic_size, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_line_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto aabb_pred = [&query](const auto &aabb) {
              return tf::intersects(query, aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto aabb_pred = [&query, threshold2](const auto &aabb) {
              return tf::distance2(
                  tf::make_sphere(aabb.center(), aabb.diagonal().length() / 2), query) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // ============================================================================
  // gather_ids - int64, float, ngon=3, 3D
  // ============================================================================

  // Point - int643float3d
  m.def("gather_ids_point_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_point_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Segment - int643float3d
  m.def("gather_ids_segment_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_segment_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Polygon - int643float3d
  m.def("gather_ids_polygon_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<-1, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_polygon_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Ray - int643float3d
  m.def("gather_ids_ray_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_ray_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto aabb_pred = [&query](const auto &aabb) {
              return tf::intersects(query, aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto aabb_pred = [&query, threshold2](const auto &aabb) {
              return tf::distance2(
                  tf::make_sphere(aabb.center(), aabb.diagonal().length() / 2), query) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Line - int643float3d
  m.def("gather_ids_line_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_line_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto aabb_pred = [&query](const auto &aabb) {
              return tf::intersects(query, aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto aabb_pred = [&query, threshold2](const auto &aabb) {
              return tf::distance2(
                  tf::make_sphere(aabb.center(), aabb.diagonal().length() / 2), query) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // ============================================================================
  // gather_ids - int64, float, dynamic, 3D
  // ============================================================================

  // Point - int64dynfloat3d
  m.def("gather_ids_point_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_point_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Segment - int64dynfloat3d
  m.def("gather_ids_segment_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_segment_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Polygon - int64dynfloat3d
  m.def("gather_ids_polygon_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<-1, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_polygon_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Ray - int64dynfloat3d
  m.def("gather_ids_ray_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_ray_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto aabb_pred = [&query](const auto &aabb) {
              return tf::intersects(query, aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto aabb_pred = [&query, threshold2](const auto &aabb) {
              return tf::distance2(
                  tf::make_sphere(aabb.center(), aabb.diagonal().length() / 2), query) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Line - int64dynfloat3d
  m.def("gather_ids_line_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, tf::dynamic_size, 3> &mesh,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_line_from_array<3, float>(query_array);

          if (predicate_type == "intersects") {
            auto aabb_pred = [&query](const auto &aabb) {
              return tf::intersects(query, aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            float threshold2 = (*threshold) * (*threshold);
            auto aabb_pred = [&query, threshold2](const auto &aabb) {
              return tf::distance2(
                  tf::make_sphere(aabb.center(), aabb.diagonal().length() / 2), query) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<float, 3>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

}

} // namespace tf::py

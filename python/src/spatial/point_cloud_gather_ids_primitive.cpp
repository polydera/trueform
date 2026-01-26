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
#include <trueform/python/spatial/point_cloud.hpp>
#include <trueform/python/spatial/gather_ids.hpp>

namespace tf::py {

auto register_point_cloud_gather_ids_primitive(nanobind::module_ &m) -> void {

  namespace nb = nanobind;

  // ============================================================================
  // gather_ids - Point queries
  // ============================================================================

  // 2D float
  m.def("gather_ids_point_float2d",
        [](point_cloud_wrapper<float, 2> &cloud,
           nb::ndarray<nb::numpy, const float, nb::shape<2>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_point_from_array<2, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 2>(cloud, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 2D double
  m.def("gather_ids_point_double2d",
        [](point_cloud_wrapper<double, 2> &cloud,
           nb::ndarray<nb::numpy, const double, nb::shape<2>> query_array,
           const std::string &predicate_type, std::optional<double> threshold) {
          auto query = make_point_from_array<2, double>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<double, 2>(cloud, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            double threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<double, 2>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 3D float
  m.def("gather_ids_point_float3d",
        [](point_cloud_wrapper<float, 3> &cloud,
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
            return gather_ids<float, 3>(cloud, aabb_pred, prim_pred);
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
            return gather_ids<float, 3>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 3D double
  m.def("gather_ids_point_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nb::ndarray<nb::numpy, const double, nb::shape<3>> query_array,
           const std::string &predicate_type, std::optional<double> threshold) {
          auto query = make_point_from_array<3, double>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<double, 3>(cloud, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            double threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<double, 3>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // ============================================================================
  // gather_ids - Segment queries
  // ============================================================================

  // 2D float
  m.def("gather_ids_segment_float2d",
        [](point_cloud_wrapper<float, 2> &cloud,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 2>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_segment_from_array<2, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 2>(cloud, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 2D double
  m.def("gather_ids_segment_double2d",
        [](point_cloud_wrapper<double, 2> &cloud,
           nb::ndarray<nb::numpy, const double, nb::shape<2, 2>> query_array,
           const std::string &predicate_type, std::optional<double> threshold) {
          auto query = make_segment_from_array<2, double>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<double, 2>(cloud, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            double threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<double, 2>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 3D float
  m.def("gather_ids_segment_float3d",
        [](point_cloud_wrapper<float, 3> &cloud,
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
            return gather_ids<float, 3>(cloud, aabb_pred, prim_pred);
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
            return gather_ids<float, 3>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 3D double
  m.def("gather_ids_segment_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nb::ndarray<nb::numpy, const double, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<double> threshold) {
          auto query = make_segment_from_array<3, double>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<double, 3>(cloud, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            double threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<double, 3>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // ============================================================================
  // gather_ids - Polygon queries
  // ============================================================================

  // 2D float
  m.def("gather_ids_polygon_float2d",
        [](point_cloud_wrapper<float, 2> &cloud,
           nb::ndarray<nb::numpy, const float, nb::shape<-1, 2>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_polygon_from_array<2, float>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 2>(cloud, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 2D double
  m.def("gather_ids_polygon_double2d",
        [](point_cloud_wrapper<double, 2> &cloud,
           nb::ndarray<nb::numpy, const double, nb::shape<-1, 2>> query_array,
           const std::string &predicate_type, std::optional<double> threshold) {
          auto query = make_polygon_from_array<2, double>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<double, 2>(cloud, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            double threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<double, 2>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 3D float
  m.def("gather_ids_polygon_float3d",
        [](point_cloud_wrapper<float, 3> &cloud,
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
            return gather_ids<float, 3>(cloud, aabb_pred, prim_pred);
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
            return gather_ids<float, 3>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 3D double
  m.def("gather_ids_polygon_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nb::ndarray<nb::numpy, const double, nb::shape<-1, 3>> query_array,
           const std::string &predicate_type, std::optional<double> threshold) {
          auto query = make_polygon_from_array<3, double>(query_array);

          if (predicate_type == "intersects") {
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb](const auto &aabb) {
              return tf::intersects(aabb, query_aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<double, 3>(cloud, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            double threshold2 = (*threshold) * (*threshold);
            auto query_aabb = tf::aabb_from(query);
            auto aabb_pred = [query_aabb, threshold2](const auto &aabb) {
              return tf::distance2(aabb, query_aabb) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<double, 3>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // ============================================================================
  // gather_ids - Ray queries
  // ============================================================================

  // 2D float
  m.def("gather_ids_ray_float2d",
        [](point_cloud_wrapper<float, 2> &cloud,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 2>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_ray_from_array<2, float>(query_array);

          if (predicate_type == "intersects") {
            auto aabb_pred = [&query](const auto &aabb) {
              return tf::intersects(query, aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 2>(cloud, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 2D double
  m.def("gather_ids_ray_double2d",
        [](point_cloud_wrapper<double, 2> &cloud,
           nb::ndarray<nb::numpy, const double, nb::shape<2, 2>> query_array,
           const std::string &predicate_type, std::optional<double> threshold) {
          auto query = make_ray_from_array<2, double>(query_array);

          if (predicate_type == "intersects") {
            auto aabb_pred = [&query](const auto &aabb) {
              return tf::intersects(query, aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<double, 2>(cloud, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            double threshold2 = (*threshold) * (*threshold);
            auto aabb_pred = [&query, threshold2](const auto &aabb) {
              return tf::distance2(
                  tf::make_sphere(aabb.center(), aabb.diagonal().length() / 2), query) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<double, 2>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 3D float
  m.def("gather_ids_ray_float3d",
        [](point_cloud_wrapper<float, 3> &cloud,
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
            return gather_ids<float, 3>(cloud, aabb_pred, prim_pred);
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
            return gather_ids<float, 3>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 3D double
  m.def("gather_ids_ray_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nb::ndarray<nb::numpy, const double, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<double> threshold) {
          auto query = make_ray_from_array<3, double>(query_array);

          if (predicate_type == "intersects") {
            auto aabb_pred = [&query](const auto &aabb) {
              return tf::intersects(query, aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<double, 3>(cloud, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            double threshold2 = (*threshold) * (*threshold);
            auto aabb_pred = [&query, threshold2](const auto &aabb) {
              return tf::distance2(
                  tf::make_sphere(aabb.center(), aabb.diagonal().length() / 2), query) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<double, 3>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // ============================================================================
  // gather_ids - Line queries
  // ============================================================================

  // 2D float
  m.def("gather_ids_line_float2d",
        [](point_cloud_wrapper<float, 2> &cloud,
           nb::ndarray<nb::numpy, const float, nb::shape<2, 2>> query_array,
           const std::string &predicate_type, std::optional<float> threshold) {
          auto query = make_line_from_array<2, float>(query_array);

          if (predicate_type == "intersects") {
            auto aabb_pred = [&query](const auto &aabb) {
              return tf::intersects(query, aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<float, 2>(cloud, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 2D double
  m.def("gather_ids_line_double2d",
        [](point_cloud_wrapper<double, 2> &cloud,
           nb::ndarray<nb::numpy, const double, nb::shape<2, 2>> query_array,
           const std::string &predicate_type, std::optional<double> threshold) {
          auto query = make_line_from_array<2, double>(query_array);

          if (predicate_type == "intersects") {
            auto aabb_pred = [&query](const auto &aabb) {
              return tf::intersects(query, aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<double, 2>(cloud, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            double threshold2 = (*threshold) * (*threshold);
            auto aabb_pred = [&query, threshold2](const auto &aabb) {
              return tf::distance2(
                  tf::make_sphere(aabb.center(), aabb.diagonal().length() / 2), query) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<double, 2>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 3D float
  m.def("gather_ids_line_float3d",
        [](point_cloud_wrapper<float, 3> &cloud,
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
            return gather_ids<float, 3>(cloud, aabb_pred, prim_pred);
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
            return gather_ids<float, 3>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // 3D double
  m.def("gather_ids_line_double3d",
        [](point_cloud_wrapper<double, 3> &cloud,
           nb::ndarray<nb::numpy, const double, nb::shape<2, 3>> query_array,
           const std::string &predicate_type, std::optional<double> threshold) {
          auto query = make_line_from_array<3, double>(query_array);

          if (predicate_type == "intersects") {
            auto aabb_pred = [&query](const auto &aabb) {
              return tf::intersects(query, aabb);
            };
            auto prim_pred = [&query](const auto &prim) {
              return tf::intersects(prim, query);
            };
            return gather_ids<double, 3>(cloud, aabb_pred, prim_pred);
          } else if (predicate_type == "within_distance") {
            if (!threshold)
              throw std::runtime_error("threshold required for within_distance");
            double threshold2 = (*threshold) * (*threshold);
            auto aabb_pred = [&query, threshold2](const auto &aabb) {
              return tf::distance2(
                  tf::make_sphere(aabb.center(), aabb.diagonal().length() / 2), query) <= threshold2;
            };
            auto prim_pred = [&query, threshold2](const auto &prim) {
              return tf::distance2(prim, query) <= threshold2;
            };
            return gather_ids<double, 3>(cloud, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("cloud"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

}

} // namespace tf::py

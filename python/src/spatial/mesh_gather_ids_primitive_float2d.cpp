/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
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

auto register_mesh_gather_ids_primitive_float2d(nanobind::module_ &m) -> void {

  namespace nb = nanobind;

  // ============================================================================
  // gather_ids - int, float, ngon=3, 2D
  // ============================================================================

  // Point - intfloat32d
  m.def("gather_ids_point_intfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Segment - intfloat32d
  m.def("gather_ids_segment_intfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Polygon - intfloat32d
  m.def("gather_ids_polygon_intfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Ray - intfloat32d
  m.def("gather_ids_ray_intfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Line - intfloat32d
  m.def("gather_ids_line_intfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // ============================================================================
  // gather_ids - int, float, ngon=4, 2D
  // ============================================================================

  // Point - intfloat42d
  m.def("gather_ids_point_intfloat42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Segment - intfloat42d
  m.def("gather_ids_segment_intfloat42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Polygon - intfloat42d
  m.def("gather_ids_polygon_intfloat42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Ray - intfloat42d
  m.def("gather_ids_ray_intfloat42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Line - intfloat42d
  m.def("gather_ids_line_intfloat42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // ============================================================================
  // gather_ids - int64, float, ngon=3, 2D
  // ============================================================================

  // Point - int64float32d
  m.def("gather_ids_point_int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Segment - int64float32d
  m.def("gather_ids_segment_int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Polygon - int64float32d
  m.def("gather_ids_polygon_int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Ray - int64float32d
  m.def("gather_ids_ray_int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Line - int64float32d
  m.def("gather_ids_line_int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // ============================================================================
  // gather_ids - int64, float, ngon=4, 2D
  // ============================================================================

  // Point - int64float42d
  m.def("gather_ids_point_int64float42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Segment - int64float42d
  m.def("gather_ids_segment_int64float42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Polygon - int64float42d
  m.def("gather_ids_polygon_int64float42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Ray - int64float42d
  m.def("gather_ids_ray_int64float42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

  // Line - int64float42d
  m.def("gather_ids_line_int64float42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
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
            return gather_ids<float, 2>(mesh, aabb_pred, prim_pred);
          } else {
            throw std::runtime_error("Unknown predicate: " + predicate_type);
          }
        },
        nb::arg("mesh"), nb::arg("query"), nb::arg("predicate_type"),
        nb::arg("threshold").none() = nb::none());

}

} // namespace tf::py

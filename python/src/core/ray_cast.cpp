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

#include "trueform/python/core/ray_cast.hpp"
#include "trueform/python/core/make_primitives.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/ray_cast.hpp>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>
#include <trueform/core/ray_config.hpp>
#include <trueform/python/util/ray_config_helper.hpp>
#include <optional>
#include <tuple>

namespace tf::py {

namespace {

// Convert ray_cast_info result to Python Optional[float]
template <typename RealT, typename ResultT>
auto ray_cast_info_to_optional(const ResultT &result) {
  if (result) {
    // Intersection occurred, return t
    return nanobind::cast(result.t);
  } else {
    // No intersection, return None
    return nanobind::none();
  }
}

} // anonymous namespace

auto register_core_ray_cast(nanobind::module_ &m) -> void {
  // ==== Ray to Plane (3D only) ====
  // Ray to Plane (float, 3D)
  m.def(
      "ray_cast_ray_plane_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
             plane_data,
           std::optional<std::tuple<float, float>> opt_config) {
        auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto plane = make_plane_from_array<3, float>(plane_data);
        return ray_cast_info_to_optional<float>(tf::ray_cast(ray, plane, config));
      },
      nanobind::arg("ray"), nanobind::arg("plane"),
      nanobind::arg("config").none() = nanobind::none());

  // Ray to Plane (double, 3D)
  m.def(
      "ray_cast_ray_plane_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
             plane_data,
           std::optional<std::tuple<double, double>> opt_config) {
        auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto plane = make_plane_from_array<3, double>(plane_data);
        return ray_cast_info_to_optional<double>(tf::ray_cast(ray, plane, config));
      },
      nanobind::arg("ray"), nanobind::arg("plane"),
      nanobind::arg("config").none() = nanobind::none());

  // ==== Ray to Polygon ====
  // Ray to Polygon (float, 2D)
  m.def(
      "ray_cast_ray_polygon_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float> poly_data,
           std::optional<std::tuple<float, float>> opt_config) {
          auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<2, float>(ray_data);
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return ray_cast_info_to_optional<float>(tf::ray_cast(ray, poly, config));
        },
      nanobind::arg("ray"), nanobind::arg("polygon"),
      nanobind::arg("config").none() = nanobind::none());

  // Ray to Polygon (float, 3D)
  m.def(
      "ray_cast_ray_polygon_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const float> poly_data,
           std::optional<std::tuple<float, float>> opt_config) {
          auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<3, float>(ray_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return ray_cast_info_to_optional<float>(tf::ray_cast(ray, poly, config));
        },
      nanobind::arg("ray"), nanobind::arg("polygon"),
      nanobind::arg("config").none() = nanobind::none());

  // Ray to Polygon (double, 2D)
  m.def(
      "ray_cast_ray_polygon_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const double> poly_data,
           std::optional<std::tuple<double, double>> opt_config) {
          auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<2, double>(ray_data);
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return ray_cast_info_to_optional<double>(tf::ray_cast(ray, poly, config));
        },
      nanobind::arg("ray"), nanobind::arg("polygon"),
      nanobind::arg("config").none() = nanobind::none());

  // Ray to Polygon (double, 3D)
  m.def(
      "ray_cast_ray_polygon_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
               ray_data,
           nanobind::ndarray<nanobind::numpy, const double> poly_data,
           std::optional<std::tuple<double, double>> opt_config) {
          auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<3, double>(ray_data);
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return ray_cast_info_to_optional<double>(tf::ray_cast(ray, poly, config));
        },
      nanobind::arg("ray"), nanobind::arg("polygon"),
      nanobind::arg("config").none() = nanobind::none());

  // ==== Ray to Segment ====
  // Ray to Segment (float, 2D)
  m.def(
      "ray_cast_ray_segment_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             seg_data,
           std::optional<std::tuple<float, float>> opt_config) {
        auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto seg = make_segment_from_array<2, float>(seg_data);
        return ray_cast_info_to_optional<float>(tf::ray_cast(ray, seg, config));
      },
      nanobind::arg("ray"), nanobind::arg("segment"),
      nanobind::arg("config").none() = nanobind::none());

  // Ray to Segment (float, 3D)
  m.def(
      "ray_cast_ray_segment_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             seg_data,
           std::optional<std::tuple<float, float>> opt_config) {
        auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto seg = make_segment_from_array<3, float>(seg_data);
        return ray_cast_info_to_optional<float>(tf::ray_cast(ray, seg, config));
      },
      nanobind::arg("ray"), nanobind::arg("segment"),
      nanobind::arg("config").none() = nanobind::none());

  // Ray to Segment (double, 2D)
  m.def(
      "ray_cast_ray_segment_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             seg_data,
           std::optional<std::tuple<double, double>> opt_config) {
        auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto seg = make_segment_from_array<2, double>(seg_data);
        return ray_cast_info_to_optional<double>(tf::ray_cast(ray, seg, config));
      },
      nanobind::arg("ray"), nanobind::arg("segment"),
      nanobind::arg("config").none() = nanobind::none());

  // Ray to Segment (double, 3D)
  m.def(
      "ray_cast_ray_segment_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             seg_data,
           std::optional<std::tuple<double, double>> opt_config) {
        auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto seg = make_segment_from_array<3, double>(seg_data);
        return ray_cast_info_to_optional<double>(tf::ray_cast(ray, seg, config));
      },
      nanobind::arg("ray"), nanobind::arg("segment"),
      nanobind::arg("config").none() = nanobind::none());

  // ==== Ray to Line ====
  // Ray to Line (float, 2D)
  m.def(
      "ray_cast_ray_line_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             line_data,
           std::optional<std::tuple<float, float>> opt_config) {
        auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto line = make_line_from_array<2, float>(line_data);
        return ray_cast_info_to_optional<float>(tf::ray_cast(ray, line, config));
      },
      nanobind::arg("ray"), nanobind::arg("line"),
      nanobind::arg("config").none() = nanobind::none());

  // Ray to Line (float, 3D)
  m.def(
      "ray_cast_ray_line_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             line_data,
           std::optional<std::tuple<float, float>> opt_config) {
        auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto line = make_line_from_array<3, float>(line_data);
        return ray_cast_info_to_optional<float>(tf::ray_cast(ray, line, config));
      },
      nanobind::arg("ray"), nanobind::arg("line"),
      nanobind::arg("config").none() = nanobind::none());

  // Ray to Line (double, 2D)
  m.def(
      "ray_cast_ray_line_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             line_data,
           std::optional<std::tuple<double, double>> opt_config) {
        auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto line = make_line_from_array<2, double>(line_data);
        return ray_cast_info_to_optional<double>(tf::ray_cast(ray, line, config));
      },
      nanobind::arg("ray"), nanobind::arg("line"),
      nanobind::arg("config").none() = nanobind::none());

  // Ray to Line (double, 3D)
  m.def(
      "ray_cast_ray_line_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             line_data,
           std::optional<std::tuple<double, double>> opt_config) {
        auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto line = make_line_from_array<3, double>(line_data);
        return ray_cast_info_to_optional<double>(tf::ray_cast(ray, line, config));
      },
      nanobind::arg("ray"), nanobind::arg("line"),
      nanobind::arg("config").none() = nanobind::none());

  // ==== Ray to AABB ====
  // Ray to AABB (float, 2D)
  m.def(
      "ray_cast_ray_aabb_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             aabb_data,
           std::optional<std::tuple<float, float>> opt_config) {
        auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto aabb = make_aabb_from_array<2, float>(aabb_data);
        return ray_cast_info_to_optional<float>(tf::ray_cast(ray, aabb, config));
      },
      nanobind::arg("ray"), nanobind::arg("aabb"),
      nanobind::arg("config").none() = nanobind::none());

  // Ray to AABB (float, 3D)
  m.def(
      "ray_cast_ray_aabb_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             aabb_data,
           std::optional<std::tuple<float, float>> opt_config) {
        auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto aabb = make_aabb_from_array<3, float>(aabb_data);
        return ray_cast_info_to_optional<float>(tf::ray_cast(ray, aabb, config));
      },
      nanobind::arg("ray"), nanobind::arg("aabb"),
      nanobind::arg("config").none() = nanobind::none());

  // Ray to AABB (double, 2D)
  m.def(
      "ray_cast_ray_aabb_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             aabb_data,
           std::optional<std::tuple<double, double>> opt_config) {
        auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto aabb = make_aabb_from_array<2, double>(aabb_data);
        return ray_cast_info_to_optional<double>(tf::ray_cast(ray, aabb, config));
      },
      nanobind::arg("ray"), nanobind::arg("aabb"),
      nanobind::arg("config").none() = nanobind::none());

  // Ray to AABB (double, 3D)
  m.def(
      "ray_cast_ray_aabb_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             aabb_data,
           std::optional<std::tuple<double, double>> opt_config) {
        auto config = tf::py::make_ray_config_from_optional(opt_config);
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto aabb = make_aabb_from_array<3, double>(aabb_data);
        return ray_cast_info_to_optional<double>(tf::ray_cast(ray, aabb, config));
      },
      nanobind::arg("ray"), nanobind::arg("aabb"),
      nanobind::arg("config").none() = nanobind::none());
}

} // namespace tf::py

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
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <trueform/python/core/make_primitives.hpp>
#include <tuple>
#include <trueform/python/spatial/point_cloud.hpp>
#include <trueform/python/spatial/ray_cast.hpp>

namespace tf::py {

auto register_point_cloud_ray_cast(nanobind::module_ &m) -> void {

  // ============================================================================
  // Ray cast on point clouds
  // ============================================================================

  // 2D float
  m.def(
      "ray_cast_point_cloud_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         point_cloud_wrapper<float, 2> &cloud,
         std::optional<std::tuple<float, float>> config) {
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto result = ray_cast(ray, cloud, config);

        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("cloud"),
      nanobind::arg("config").none() = nanobind::none());

  // 2D double
  m.def(
      "ray_cast_point_cloud_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         point_cloud_wrapper<double, 2> &cloud,
         std::optional<std::tuple<double, double>> config) {
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto result = ray_cast(ray, cloud, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("cloud"),
      nanobind::arg("config").none() = nanobind::none());

  // 3D float
  m.def(
      "ray_cast_point_cloud_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         point_cloud_wrapper<float, 3> &cloud,
         std::optional<std::tuple<float, float>> config) {
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto result = ray_cast(ray, cloud, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::object(nanobind::none());
        }
      },
      nanobind::arg("ray"), nanobind::arg("cloud"),
      nanobind::arg("config").none() = nanobind::none());

  // 3D double
  m.def(
      "ray_cast_point_cloud_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         point_cloud_wrapper<double, 3> &cloud,
         std::optional<std::tuple<double, double>> config) {
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto result = ray_cast(ray, cloud, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::object(nanobind::none());
        }
      },
      nanobind::arg("ray"), nanobind::arg("cloud"),
      nanobind::arg("config").none() = nanobind::none());
}

} // namespace tf::py

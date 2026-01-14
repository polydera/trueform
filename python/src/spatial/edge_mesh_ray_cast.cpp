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
#include <trueform/python/spatial/edge_mesh.hpp>
#include <trueform/python/core/make_primitives.hpp>
#include <trueform/python/spatial/ray_cast.hpp>
#include <tuple>

namespace tf::py {

auto register_edge_mesh_ray_cast(nanobind::module_ &m) -> void {

  // ============================================================================
  // Ray cast on edge meshes - all type combinations
  // Index types: int, int64
  // Real types: float, double
  // Dims: 2D, 3D
  // Total: 8 combinations
  // ============================================================================

  // int32, float, 2D
  m.def(
      "ray_cast_edge_mesh_intfloat2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         edge_mesh_wrapper<int, float, 2> &edge_mesh,
         std::optional<std::pair<float, float>> config) {
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto result = ray_cast(ray, edge_mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("edge_mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int32, float, 3D
  m.def(
      "ray_cast_edge_mesh_intfloat3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         edge_mesh_wrapper<int, float, 3> &edge_mesh,
         std::optional<std::pair<float, float>> config) {
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto result = ray_cast(ray, edge_mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("edge_mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int32, double, 2D
  m.def(
      "ray_cast_edge_mesh_intdouble2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         edge_mesh_wrapper<int, double, 2> &edge_mesh,
         std::optional<std::pair<double, double>> config) {
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto result = ray_cast(ray, edge_mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("edge_mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int32, double, 3D
  m.def(
      "ray_cast_edge_mesh_intdouble3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         edge_mesh_wrapper<int, double, 3> &edge_mesh,
         std::optional<std::pair<double, double>> config) {
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto result = ray_cast(ray, edge_mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("edge_mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int64, float, 2D
  m.def(
      "ray_cast_edge_mesh_int64float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
         std::optional<std::pair<float, float>> config) {
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto result = ray_cast(ray, edge_mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("edge_mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int64, float, 3D
  m.def(
      "ray_cast_edge_mesh_int64float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
         std::optional<std::pair<float, float>> config) {
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto result = ray_cast(ray, edge_mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("edge_mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int64, double, 2D
  m.def(
      "ray_cast_edge_mesh_int64double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
         std::optional<std::pair<double, double>> config) {
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto result = ray_cast(ray, edge_mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("edge_mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int64, double, 3D
  m.def(
      "ray_cast_edge_mesh_int64double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
         std::optional<std::pair<double, double>> config) {
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto result = ray_cast(ray, edge_mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("edge_mesh"),
      nanobind::arg("config").none() = nanobind::none());
}

} // namespace tf::py

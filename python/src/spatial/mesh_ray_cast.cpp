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
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/ray_cast.hpp>
#include <tuple>

namespace tf::py {

auto register_mesh_ray_cast(nanobind::module_ &m) -> void {

  // ============================================================================
  // Ray cast on meshes - all type combinations
  // Index types: int, int64
  // Real types: float, double
  // Ngon: 3 (triangles), dynamic (variable-size)
  // Dims: 2D, 3D
  // Total: 16 combinations
  // ============================================================================

  // int32, float, triangle, 2D
  m.def(
      "ray_cast_mesh_int3float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int, float, 3, 2> &mesh,
         std::optional<std::tuple<float, float>> config) {
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int32, float, triangle, 3D
  m.def(
      "ray_cast_mesh_int3float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int, float, 3, 3> &mesh,
         std::optional<std::tuple<float, float>> config) {
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int32, float, dynamic, 2D
  m.def(
      "ray_cast_mesh_intdynfloat2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int, float, dynamic_size, 2> &mesh,
         std::optional<std::tuple<float, float>> config) {
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int32, float, dynamic, 3D
  m.def(
      "ray_cast_mesh_intdynfloat3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int, float, dynamic_size, 3> &mesh,
         std::optional<std::tuple<float, float>> config) {
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int32, double, triangle, 2D
  m.def(
      "ray_cast_mesh_int3double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int, double, 3, 2> &mesh,
         std::optional<std::tuple<double, double>> config) {
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int32, double, triangle, 3D
  m.def(
      "ray_cast_mesh_int3double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int, double, 3, 3> &mesh,
         std::optional<std::tuple<double, double>> config) {
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int32, double, dynamic, 2D
  m.def(
      "ray_cast_mesh_intdyndouble2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int, double, dynamic_size, 2> &mesh,
         std::optional<std::tuple<double, double>> config) {
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int32, double, dynamic, 3D
  m.def(
      "ray_cast_mesh_intdyndouble3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int, double, dynamic_size, 3> &mesh,
         std::optional<std::tuple<double, double>> config) {
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int64, float, triangle, 2D
  m.def(
      "ray_cast_mesh_int643float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int64_t, float, 3, 2> &mesh,
         std::optional<std::tuple<float, float>> config) {
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int64, float, triangle, 3D
  m.def(
      "ray_cast_mesh_int643float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int64_t, float, 3, 3> &mesh,
         std::optional<std::tuple<float, float>> config) {
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int64, float, dynamic, 2D
  m.def(
      "ray_cast_mesh_int64dynfloat2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int64_t, float, dynamic_size, 2> &mesh,
         std::optional<std::tuple<float, float>> config) {
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int64, float, dynamic, 3D
  m.def(
      "ray_cast_mesh_int64dynfloat3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
         std::optional<std::tuple<float, float>> config) {
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int64, double, triangle, 2D
  m.def(
      "ray_cast_mesh_int643double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int64_t, double, 3, 2> &mesh,
         std::optional<std::tuple<double, double>> config) {
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int64, double, triangle, 3D
  m.def(
      "ray_cast_mesh_int643double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int64_t, double, 3, 3> &mesh,
         std::optional<std::tuple<double, double>> config) {
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int64, double, dynamic, 2D
  m.def(
      "ray_cast_mesh_int64dyndouble2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int64_t, double, dynamic_size, 2> &mesh,
         std::optional<std::tuple<double, double>> config) {
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());

  // int64, double, dynamic, 3D
  m.def(
      "ray_cast_mesh_int64dyndouble3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
         std::optional<std::tuple<double, double>> config) {
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto result = ray_cast(ray, mesh, config);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"),
      nanobind::arg("config").none() = nanobind::none());
}

} // namespace tf::py

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <trueform/python/core/make_primitives.hpp>
#include <trueform/python/core/mesh.hpp>
#include <trueform/python/spatial/ray_cast.hpp>

namespace tf::py {

auto register_mesh_ray_cast(nanobind::module_ &m) -> void {

  // ============================================================================
  // Ray cast on meshes - all type combinations
  // Index types: int, int64
  // Real types: float, double
  // Ngon: 3 (triangles), 4 (quads)
  // Dims: 2D, 3D
  // Total: 16 combinations
  // ============================================================================

  // int32, float, triangle, 2D
  m.def(
      "ray_cast_mesh_intfloat32d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int, float, 3, 2> &mesh) {
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int32, float, triangle, 3D
  m.def(
      "ray_cast_mesh_intfloat33d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int, float, 3, 3> &mesh) {
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int32, float, quad, 2D
  m.def(
      "ray_cast_mesh_intfloat42d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int, float, 4, 2> &mesh) {
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int32, float, quad, 3D
  m.def(
      "ray_cast_mesh_intfloat43d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int, float, 4, 3> &mesh) {
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int32, double, triangle, 2D
  m.def(
      "ray_cast_mesh_intdouble32d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int, double, 3, 2> &mesh) {
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int32, double, triangle, 3D
  m.def(
      "ray_cast_mesh_intdouble33d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int, double, 3, 3> &mesh) {
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int32, double, quad, 2D
  m.def(
      "ray_cast_mesh_intdouble42d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int, double, 4, 2> &mesh) {
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int32, double, quad, 3D
  m.def(
      "ray_cast_mesh_intdouble43d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int, double, 4, 3> &mesh) {
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int64, float, triangle, 2D
  m.def(
      "ray_cast_mesh_int64float32d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int64_t, float, 3, 2> &mesh) {
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int64, float, triangle, 3D
  m.def(
      "ray_cast_mesh_int64float33d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int64_t, float, 3, 3> &mesh) {
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int64, float, quad, 2D
  m.def(
      "ray_cast_mesh_int64float42d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int64_t, float, 4, 2> &mesh) {
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int64, float, quad, 3D
  m.def(
      "ray_cast_mesh_int64float43d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int64_t, float, 4, 3> &mesh) {
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int64, double, triangle, 2D
  m.def(
      "ray_cast_mesh_int64double32d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int64_t, double, 3, 2> &mesh) {
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int64, double, triangle, 3D
  m.def(
      "ray_cast_mesh_int64double33d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int64_t, double, 3, 3> &mesh) {
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int64, double, quad, 2D
  m.def(
      "ray_cast_mesh_int64double42d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         mesh_wrapper<int64_t, double, 4, 2> &mesh) {
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));

  // int64, double, quad, 3D
  m.def(
      "ray_cast_mesh_int64double43d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         mesh_wrapper<int64_t, double, 4, 3> &mesh) {
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto result = ray_cast(ray, mesh);
        if (result) {
          return nanobind::cast(
              nanobind::make_tuple(result->first, result->second));
        } else {
          return nanobind::none();
        }
      },
      nanobind::arg("ray"), nanobind::arg("mesh"));
}

} // namespace tf::py

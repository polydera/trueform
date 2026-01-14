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

#include "trueform/python/cut/isobands_impl.hpp"

namespace tf::py {

auto register_cut_isobands(nanobind::module_ &m) -> void {
  // Bindings for all mesh variants (8 total: 2 index × 2 real × 2 dims)
  // Each has 2 versions: without curves and with curves

  // ===== int32, float32, triangles, 3D =====
  m.def("make_isobands_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_impl<int, float, 3, 3>(mesh, scalars, cut_values,
                                                       selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  m.def("make_isobands_curves_int3float3d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_with_curves_impl<int, float, 3, 3>(
              mesh, scalars, cut_values, selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  // ===== int32, float32, dynamic, 3D =====
  m.def("make_isobands_intdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_impl<int, float, dynamic_size, 3>(mesh, scalars, cut_values,
                                                       selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  m.def("make_isobands_curves_intdynfloat3d",
        [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_with_curves_impl<int, float, dynamic_size, 3>(
              mesh, scalars, cut_values, selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  // ===== int32, float64, triangles, 3D =====
  m.def("make_isobands_int3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_impl<int, double, 3, 3>(mesh, scalars, cut_values,
                                                        selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  m.def("make_isobands_curves_int3double3d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_with_curves_impl<int, double, 3, 3>(
              mesh, scalars, cut_values, selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  // ===== int32, float64, dynamic, 3D =====
  m.def("make_isobands_intdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_impl<int, double, dynamic_size, 3>(mesh, scalars, cut_values,
                                                        selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  m.def("make_isobands_curves_intdyndouble3d",
        [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_with_curves_impl<int, double, dynamic_size, 3>(
              mesh, scalars, cut_values, selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  // ===== int64, float32, triangles, 3D =====
  m.def("make_isobands_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_impl<int64_t, float, 3, 3>(
              mesh, scalars, cut_values, selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  m.def("make_isobands_curves_int643float3d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_with_curves_impl<int64_t, float, 3, 3>(
              mesh, scalars, cut_values, selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  // ===== int64, float32, dynamic, 3D =====
  m.def("make_isobands_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_impl<int64_t, float, dynamic_size, 3>(
              mesh, scalars, cut_values, selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  m.def("make_isobands_curves_int64dynfloat3d",
        [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_with_curves_impl<int64_t, float, dynamic_size, 3>(
              mesh, scalars, cut_values, selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  // ===== int64, float64, triangles, 3D =====
  m.def("make_isobands_int643double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_impl<int64_t, double, 3, 3>(
              mesh, scalars, cut_values, selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  m.def("make_isobands_curves_int643double3d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_with_curves_impl<int64_t, double, 3, 3>(
              mesh, scalars, cut_values, selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  // ===== int64, float64, dynamic, 3D =====
  m.def("make_isobands_int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_impl<int64_t, double, dynamic_size, 3>(
              mesh, scalars, cut_values, selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  m.def("make_isobands_curves_int64dyndouble3d",
        [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               scalars,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
               cut_values,
           nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
               selected_bands) {
          return make_isobands_with_curves_impl<int64_t, double, dynamic_size, 3>(
              mesh, scalars, cut_values, selected_bands);
        },
        nanobind::arg("mesh"), nanobind::arg("scalars"),
        nanobind::arg("cut_values"), nanobind::arg("selected_bands"));

  // TODO: Add 2D variants if needed (int3float2d, etc.)
}

} // namespace tf::py

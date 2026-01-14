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

#include "trueform/python/intersect/isocontours.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_intersect_isocontours(nanobind::module_ &m) -> void {
  // ==== 3D Triangle Meshes ====

  // int32, float, triangles, 3D
  m.def(
      "make_isocontours_single_int3float3d",
      [](mesh_wrapper<int, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         float threshold) {
        return make_isocontours_single_impl<int, float, 3, 3>(mesh, scalars,
                                                            threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_int3float3d",
      [](mesh_wrapper<int, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int, float, 3, 3>(mesh, scalars,
                                                           thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));

  // int32, float, dynamic, 3D
  m.def(
      "make_isocontours_single_intdynfloat3d",
      [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         float threshold) {
        return make_isocontours_single_impl<int, float, dynamic_size, 3>(mesh, scalars,
                                                            threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_intdynfloat3d",
      [](mesh_wrapper<int, float, dynamic_size, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int, float, dynamic_size, 3>(mesh, scalars,
                                                           thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));

  // int32, double, triangles, 3D
  m.def(
      "make_isocontours_single_int3double3d",
      [](mesh_wrapper<int, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         double threshold) {
        return make_isocontours_single_impl<int, double, 3, 3>(mesh, scalars,
                                                             threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_int3double3d",
      [](mesh_wrapper<int, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int, double, 3, 3>(mesh, scalars,
                                                            thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));

  // int32, double, dynamic, 3D
  m.def(
      "make_isocontours_single_intdyndouble3d",
      [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         double threshold) {
        return make_isocontours_single_impl<int, double, dynamic_size, 3>(mesh, scalars,
                                                             threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_intdyndouble3d",
      [](mesh_wrapper<int, double, dynamic_size, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int, double, dynamic_size, 3>(mesh, scalars,
                                                            thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));

  // int64, float, triangles, 3D
  m.def(
      "make_isocontours_single_int643float3d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         float threshold) {
        return make_isocontours_single_impl<int64_t, float, 3, 3>(mesh, scalars,
                                                                threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_int643float3d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int64_t, float, 3, 3>(mesh, scalars,
                                                               thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));

  // int64, float, dynamic, 3D
  m.def(
      "make_isocontours_single_int64dynfloat3d",
      [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         float threshold) {
        return make_isocontours_single_impl<int64_t, float, dynamic_size, 3>(mesh, scalars,
                                                                threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_int64dynfloat3d",
      [](mesh_wrapper<int64_t, float, dynamic_size, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int64_t, float, dynamic_size, 3>(mesh, scalars,
                                                               thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));

  // int64, double, triangles, 3D
  m.def(
      "make_isocontours_single_int643double3d",
      [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         double threshold) {
        return make_isocontours_single_impl<int64_t, double, 3, 3>(mesh, scalars,
                                                                 threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_int643double3d",
      [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int64_t, double, 3, 3>(mesh, scalars,
                                                                thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));

  // int64, double, dynamic, 3D
  m.def(
      "make_isocontours_single_int64dyndouble3d",
      [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         double threshold) {
        return make_isocontours_single_impl<int64_t, double, dynamic_size, 3>(mesh, scalars,
                                                                 threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_int64dyndouble3d",
      [](mesh_wrapper<int64_t, double, dynamic_size, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int64_t, double, dynamic_size, 3>(mesh, scalars,
                                                                thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));
}

} // namespace tf::py

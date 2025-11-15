/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/intersect/isocontours.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_intersect_isocontours(nanobind::module_ &m) -> void {
  // ==== 3D Triangle Meshes ====

  // int32, float, 3D
  m.def(
      "make_isocontours_single_intfloat33d",
      [](mesh_wrapper<int, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         float threshold) {
        return make_isocontours_single_impl<int, float, 3>(mesh, scalars,
                                                            threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_intfloat33d",
      [](mesh_wrapper<int, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int, float, 3>(mesh, scalars,
                                                           thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));

  // int32, double, 3D
  m.def(
      "make_isocontours_single_intdouble33d",
      [](mesh_wrapper<int, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         double threshold) {
        return make_isocontours_single_impl<int, double, 3>(mesh, scalars,
                                                             threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_intdouble33d",
      [](mesh_wrapper<int, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int, double, 3>(mesh, scalars,
                                                            thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));

  // int64, float, 3D
  m.def(
      "make_isocontours_single_int64float33d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         float threshold) {
        return make_isocontours_single_impl<int64_t, float, 3>(mesh, scalars,
                                                                threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_int64float33d",
      [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int64_t, float, 3>(mesh, scalars,
                                                               thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));

  // int64, double, 3D
  m.def(
      "make_isocontours_single_int64double33d",
      [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         double threshold) {
        return make_isocontours_single_impl<int64_t, double, 3>(mesh, scalars,
                                                                 threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_int64double33d",
      [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int64_t, double, 3>(mesh, scalars,
                                                                thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));

  // ==== 2D Triangle Meshes ====

  // int32, float, 2D
  m.def(
      "make_isocontours_single_intfloat32d",
      [](mesh_wrapper<int, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         float threshold) {
        return make_isocontours_single_impl<int, float, 2>(mesh, scalars,
                                                            threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_intfloat32d",
      [](mesh_wrapper<int, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int, float, 2>(mesh, scalars,
                                                           thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));

  // int32, double, 2D
  m.def(
      "make_isocontours_single_intdouble32d",
      [](mesh_wrapper<int, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         double threshold) {
        return make_isocontours_single_impl<int, double, 2>(mesh, scalars,
                                                             threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_intdouble32d",
      [](mesh_wrapper<int, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int, double, 2>(mesh, scalars,
                                                            thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));

  // int64, float, 2D
  m.def(
      "make_isocontours_single_int64float32d",
      [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         float threshold) {
        return make_isocontours_single_impl<int64_t, float, 2>(mesh, scalars,
                                                                threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_int64float32d",
      [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int64_t, float, 2>(mesh, scalars,
                                                               thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));

  // int64, double, 2D
  m.def(
      "make_isocontours_single_int64double32d",
      [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         double threshold) {
        return make_isocontours_single_impl<int64_t, double, 2>(mesh, scalars,
                                                                 threshold);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("threshold"));

  m.def(
      "make_isocontours_multi_int64double32d",
      [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             scalars,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<-1>>
             thresholds) {
        return make_isocontours_multi_impl<int64_t, double, 2>(mesh, scalars,
                                                                thresholds);
      },
      nanobind::arg("mesh"), nanobind::arg("scalars"),
      nanobind::arg("thresholds"));
}

} // namespace tf::py

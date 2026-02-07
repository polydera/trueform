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
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <trueform/python/geometry/fit_knn_alignment.hpp>

namespace tf::py {

auto register_fit_knn_alignment(nanobind::module_ &m) -> void {

  // ============================================================
  // Point-to-point alignment
  // ============================================================

  // float, 2D
  m.def("fit_knn_alignment_float2d",
        [](point_cloud_wrapper<float, 2> &cloud0,
           point_cloud_wrapper<float, 2> &cloud1, std::size_t k,
           std::optional<float> sigma, float outlier_proportion) {
          return fit_knn_alignment_impl(cloud0, cloud1, k, sigma,
                                        outlier_proportion);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("k") = 1,
        nanobind::arg("sigma").none() = nanobind::none(),
        nanobind::arg("outlier_proportion") = 0.0f,
        "Fit alignment using k-NN correspondences (one ICP iteration).\n"
        "Returns a 3x3 transformation matrix.");

  // float, 3D
  m.def("fit_knn_alignment_float3d",
        [](point_cloud_wrapper<float, 3> &cloud0,
           point_cloud_wrapper<float, 3> &cloud1, std::size_t k,
           std::optional<float> sigma, float outlier_proportion) {
          return fit_knn_alignment_impl(cloud0, cloud1, k, sigma,
                                        outlier_proportion);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("k") = 1,
        nanobind::arg("sigma").none() = nanobind::none(),
        nanobind::arg("outlier_proportion") = 0.0f,
        "Fit alignment using k-NN correspondences (one ICP iteration).\n"
        "Returns a 4x4 transformation matrix.");

  // double, 2D
  m.def("fit_knn_alignment_double2d",
        [](point_cloud_wrapper<double, 2> &cloud0,
           point_cloud_wrapper<double, 2> &cloud1, std::size_t k,
           std::optional<float> sigma, float outlier_proportion) {
          return fit_knn_alignment_impl(cloud0, cloud1, k, sigma,
                                        outlier_proportion);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("k") = 1,
        nanobind::arg("sigma").none() = nanobind::none(),
        nanobind::arg("outlier_proportion") = 0.0f,
        "Fit alignment using k-NN correspondences (one ICP iteration).\n"
        "Returns a 3x3 transformation matrix.");

  // double, 3D
  m.def("fit_knn_alignment_double3d",
        [](point_cloud_wrapper<double, 3> &cloud0,
           point_cloud_wrapper<double, 3> &cloud1, std::size_t k,
           std::optional<float> sigma, float outlier_proportion) {
          return fit_knn_alignment_impl(cloud0, cloud1, k, sigma,
                                        outlier_proportion);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("k") = 1,
        nanobind::arg("sigma").none() = nanobind::none(),
        nanobind::arg("outlier_proportion") = 0.0f,
        "Fit alignment using k-NN correspondences (one ICP iteration).\n"
        "Returns a 4x4 transformation matrix.");

  // ============================================================
  // Point-to-plane alignment (target has normals)
  // ============================================================

  // float, 3D
  m.def("fit_knn_alignment_p2plane_float3d",
        [](point_cloud_wrapper<float, 3> &cloud0,
           point_cloud_wrapper<float, 3> &cloud1,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>
               normals1,
           std::size_t k, std::optional<float> sigma,
           float outlier_proportion) {
          return fit_knn_alignment_p2plane_impl(cloud0, cloud1, normals1, k,
                                                sigma, outlier_proportion);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("normals1"), nanobind::arg("k") = 1,
        nanobind::arg("sigma").none() = nanobind::none(),
        nanobind::arg("outlier_proportion") = 0.0f,
        "Fit alignment using point-to-plane distance.\n"
        "Returns a 4x4 transformation matrix.");

  // double, 3D
  m.def("fit_knn_alignment_p2plane_double3d",
        [](point_cloud_wrapper<double, 3> &cloud0,
           point_cloud_wrapper<double, 3> &cloud1,
           nanobind::ndarray<nanobind::numpy, double, nanobind::shape<-1, 3>>
               normals1,
           std::size_t k, std::optional<float> sigma,
           float outlier_proportion) {
          return fit_knn_alignment_p2plane_impl(cloud0, cloud1, normals1, k,
                                                sigma, outlier_proportion);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("normals1"), nanobind::arg("k") = 1,
        nanobind::arg("sigma").none() = nanobind::none(),
        nanobind::arg("outlier_proportion") = 0.0f,
        "Fit alignment using point-to-plane distance.\n"
        "Returns a 4x4 transformation matrix.");

  // ============================================================
  // Normal weighting alignment (both have normals)
  // ============================================================

  // float, 3D
  m.def("fit_knn_alignment_weighted_float3d",
        [](point_cloud_wrapper<float, 3> &cloud0,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>
               normals0,
           point_cloud_wrapper<float, 3> &cloud1,
           nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, 3>>
               normals1,
           std::size_t k, std::optional<float> sigma,
           float outlier_proportion) {
          return fit_knn_alignment_weighted_impl(cloud0, normals0, cloud1,
                                                 normals1, k, sigma,
                                                 outlier_proportion);
        },
        nanobind::arg("cloud0"), nanobind::arg("normals0"),
        nanobind::arg("cloud1"), nanobind::arg("normals1"),
        nanobind::arg("k") = 1,
        nanobind::arg("sigma").none() = nanobind::none(),
        nanobind::arg("outlier_proportion") = 0.0f,
        "Fit alignment with normal weighting.\n"
        "Returns a 4x4 transformation matrix.");

  // double, 3D
  m.def("fit_knn_alignment_weighted_double3d",
        [](point_cloud_wrapper<double, 3> &cloud0,
           nanobind::ndarray<nanobind::numpy, double, nanobind::shape<-1, 3>>
               normals0,
           point_cloud_wrapper<double, 3> &cloud1,
           nanobind::ndarray<nanobind::numpy, double, nanobind::shape<-1, 3>>
               normals1,
           std::size_t k, std::optional<float> sigma,
           float outlier_proportion) {
          return fit_knn_alignment_weighted_impl(cloud0, normals0, cloud1,
                                                 normals1, k, sigma,
                                                 outlier_proportion);
        },
        nanobind::arg("cloud0"), nanobind::arg("normals0"),
        nanobind::arg("cloud1"), nanobind::arg("normals1"),
        nanobind::arg("k") = 1,
        nanobind::arg("sigma").none() = nanobind::none(),
        nanobind::arg("outlier_proportion") = 0.0f,
        "Fit alignment with normal weighting.\n"
        "Returns a 4x4 transformation matrix.");
}

} // namespace tf::py

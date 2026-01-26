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
#include <trueform/python/geometry/fit_knn_alignment.hpp>

namespace tf::py {

auto register_fit_knn_alignment(nanobind::module_ &m) -> void {

  // float, 2D
  m.def("fit_knn_alignment_float2d",
        [](point_cloud_wrapper<float, 2> &cloud0,
           point_cloud_wrapper<float, 2> &cloud1, std::size_t k,
           std::optional<float> sigma) {
          return fit_knn_alignment_impl(cloud0, cloud1, k, sigma);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("k") = 1,
        nanobind::arg("sigma").none() = nanobind::none(),
        "Fit alignment using k-NN correspondences (one ICP iteration).\n"
        "Returns a 3x3 transformation matrix.");

  // float, 3D
  m.def("fit_knn_alignment_float3d",
        [](point_cloud_wrapper<float, 3> &cloud0,
           point_cloud_wrapper<float, 3> &cloud1, std::size_t k,
           std::optional<float> sigma) {
          return fit_knn_alignment_impl(cloud0, cloud1, k, sigma);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("k") = 1,
        nanobind::arg("sigma").none() = nanobind::none(),
        "Fit alignment using k-NN correspondences (one ICP iteration).\n"
        "Returns a 4x4 transformation matrix.");

  // double, 2D
  m.def("fit_knn_alignment_double2d",
        [](point_cloud_wrapper<double, 2> &cloud0,
           point_cloud_wrapper<double, 2> &cloud1, std::size_t k,
           std::optional<double> sigma) {
          return fit_knn_alignment_impl(cloud0, cloud1, k, sigma);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("k") = 1,
        nanobind::arg("sigma").none() = nanobind::none(),
        "Fit alignment using k-NN correspondences (one ICP iteration).\n"
        "Returns a 3x3 transformation matrix.");

  // double, 3D
  m.def("fit_knn_alignment_double3d",
        [](point_cloud_wrapper<double, 3> &cloud0,
           point_cloud_wrapper<double, 3> &cloud1, std::size_t k,
           std::optional<double> sigma) {
          return fit_knn_alignment_impl(cloud0, cloud1, k, sigma);
        },
        nanobind::arg("cloud0"), nanobind::arg("cloud1"),
        nanobind::arg("k") = 1,
        nanobind::arg("sigma").none() = nanobind::none(),
        "Fit alignment using k-NN correspondences (one ICP iteration).\n"
        "Returns a 4x4 transformation matrix.");
}

} // namespace tf::py

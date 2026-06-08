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

#include "trueform/python/remesh/isotropic_remeshed_impl.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <optional>

namespace tf::py {

auto register_isotropic_remeshed(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // int32, float32, 3D
  m.def(
      "isotropic_remeshed_int3float3d",
      [](mesh_wrapper<int, float, 3, 3> &wrapper, float target_length,
         int iterations, int relaxation_iters, float min_quality,
         float lambda, bool preserve_boundary, bool use_quadric,
         bool parallel, double feature_angle, float feature_weight,
         std::optional<ndarray<numpy, const int, shape<-1>>> regions) {
        return isotropic_remeshed_impl<int, float>(
            wrapper, target_length, iterations, relaxation_iters,
            min_quality, lambda, preserve_boundary, use_quadric, parallel,
            feature_angle, feature_weight, regions);
      },
      arg("mesh"), arg("target_length"), arg("iterations") = 3,
      arg("relaxation_iters") = 3, arg("min_quality") = 0.3f,
      arg("lambda_") = 0.5f, arg("preserve_boundary") = true,
      arg("use_quadric") = false, arg("parallel") = true,
      arg("feature_angle") = -1.0, arg("feature_weight") = 100.f,
      arg("preserve_regions") = nanobind::none());

  // int32, float64, 3D
  m.def(
      "isotropic_remeshed_int3double3d",
      [](mesh_wrapper<int, double, 3, 3> &wrapper, double target_length,
         int iterations, int relaxation_iters, double min_quality,
         double lambda, bool preserve_boundary, bool use_quadric,
         bool parallel, double feature_angle, double feature_weight,
         std::optional<ndarray<numpy, const int, shape<-1>>> regions) {
        return isotropic_remeshed_impl<int, double>(
            wrapper, target_length, iterations, relaxation_iters,
            min_quality, lambda, preserve_boundary, use_quadric, parallel,
            feature_angle, feature_weight, regions);
      },
      arg("mesh"), arg("target_length"), arg("iterations") = 3,
      arg("relaxation_iters") = 3, arg("min_quality") = 0.3,
      arg("lambda_") = 0.5, arg("preserve_boundary") = true,
      arg("use_quadric") = false, arg("parallel") = true,
      arg("feature_angle") = -1.0, arg("feature_weight") = 100.0,
      arg("preserve_regions") = nanobind::none());

  // int64, float32, 3D
  m.def(
      "isotropic_remeshed_int643float3d",
      [](mesh_wrapper<int64_t, float, 3, 3> &wrapper, float target_length,
         int iterations, int relaxation_iters, float min_quality,
         float lambda, bool preserve_boundary, bool use_quadric,
         bool parallel, double feature_angle, float feature_weight,
         std::optional<ndarray<numpy, const int, shape<-1>>> regions) {
        return isotropic_remeshed_impl<int64_t, float>(
            wrapper, target_length, iterations, relaxation_iters,
            min_quality, lambda, preserve_boundary, use_quadric, parallel,
            feature_angle, feature_weight, regions);
      },
      arg("mesh"), arg("target_length"), arg("iterations") = 3,
      arg("relaxation_iters") = 3, arg("min_quality") = 0.3f,
      arg("lambda_") = 0.5f, arg("preserve_boundary") = true,
      arg("use_quadric") = false, arg("parallel") = true,
      arg("feature_angle") = -1.0, arg("feature_weight") = 100.f,
      arg("preserve_regions") = nanobind::none());

  // int64, float64, 3D
  m.def(
      "isotropic_remeshed_int643double3d",
      [](mesh_wrapper<int64_t, double, 3, 3> &wrapper, double target_length,
         int iterations, int relaxation_iters, double min_quality,
         double lambda, bool preserve_boundary, bool use_quadric,
         bool parallel, double feature_angle, double feature_weight,
         std::optional<ndarray<numpy, const int, shape<-1>>> regions) {
        return isotropic_remeshed_impl<int64_t, double>(
            wrapper, target_length, iterations, relaxation_iters,
            min_quality, lambda, preserve_boundary, use_quadric, parallel,
            feature_angle, feature_weight, regions);
      },
      arg("mesh"), arg("target_length"), arg("iterations") = 3,
      arg("relaxation_iters") = 3, arg("min_quality") = 0.3,
      arg("lambda_") = 0.5, arg("preserve_boundary") = true,
      arg("use_quadric") = false, arg("parallel") = true,
      arg("feature_angle") = -1.0, arg("feature_weight") = 100.0,
      arg("preserve_regions") = nanobind::none());
}

} // namespace tf::py

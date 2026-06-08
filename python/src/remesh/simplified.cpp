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

#include "trueform/python/remesh/simplified_impl.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <optional>

namespace tf::py {

auto register_simplified(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // int32, float32, 3D
  m.def(
      "simplified_int3float3d",
      [](mesh_wrapper<int, float, 3, 3> &wrapper, float error_rel,
         int optimize_iterations, float min_quality, bool preserve_boundary,
         double stabilizer, bool parallel, double feature_angle,
         float feature_weight, int iterations, int relaxation_iters,
         float lambda,
         std::optional<ndarray<numpy, const int, shape<-1>>> regions) {
        return simplified_impl<int, float>(
            wrapper, error_rel, optimize_iterations, min_quality,
            preserve_boundary, stabilizer, parallel, feature_angle,
            feature_weight, iterations, relaxation_iters, lambda, regions);
      },
      arg("mesh"), arg("error_rel") = 0.002f, arg("optimize_iterations") = 3,
      arg("min_quality") = 0.3f, arg("preserve_boundary") = true,
      arg("stabilizer") = 1e-3, arg("parallel") = true,
      arg("feature_angle") = -1.0, arg("feature_weight") = 100.f,
      arg("iterations") = 1, arg("relaxation_iters") = 3, arg("lambda") = 0.5f,
      arg("preserve_regions") = nanobind::none());

  // int32, float64, 3D
  m.def(
      "simplified_int3double3d",
      [](mesh_wrapper<int, double, 3, 3> &wrapper, double error_rel,
         int optimize_iterations, double min_quality, bool preserve_boundary,
         double stabilizer, bool parallel, double feature_angle,
         double feature_weight, int iterations, int relaxation_iters,
         double lambda,
         std::optional<ndarray<numpy, const int, shape<-1>>> regions) {
        return simplified_impl<int, double>(
            wrapper, error_rel, optimize_iterations, min_quality,
            preserve_boundary, stabilizer, parallel, feature_angle,
            feature_weight, iterations, relaxation_iters, lambda, regions);
      },
      arg("mesh"), arg("error_rel") = 0.002, arg("optimize_iterations") = 3,
      arg("min_quality") = 0.3, arg("preserve_boundary") = true,
      arg("stabilizer") = 1e-3, arg("parallel") = true,
      arg("feature_angle") = -1.0, arg("feature_weight") = 100.0,
      arg("iterations") = 1, arg("relaxation_iters") = 3, arg("lambda") = 0.5,
      arg("preserve_regions") = nanobind::none());

  // int64, float32, 3D
  m.def(
      "simplified_int643float3d",
      [](mesh_wrapper<int64_t, float, 3, 3> &wrapper, float error_rel,
         int optimize_iterations, float min_quality, bool preserve_boundary,
         double stabilizer, bool parallel, double feature_angle,
         float feature_weight, int iterations, int relaxation_iters,
         float lambda,
         std::optional<ndarray<numpy, const int, shape<-1>>> regions) {
        return simplified_impl<int64_t, float>(
            wrapper, error_rel, optimize_iterations, min_quality,
            preserve_boundary, stabilizer, parallel, feature_angle,
            feature_weight, iterations, relaxation_iters, lambda, regions);
      },
      arg("mesh"), arg("error_rel") = 0.002f, arg("optimize_iterations") = 3,
      arg("min_quality") = 0.3f, arg("preserve_boundary") = true,
      arg("stabilizer") = 1e-3, arg("parallel") = true,
      arg("feature_angle") = -1.0, arg("feature_weight") = 100.f,
      arg("iterations") = 1, arg("relaxation_iters") = 3, arg("lambda") = 0.5f,
      arg("preserve_regions") = nanobind::none());

  // int64, float64, 3D
  m.def(
      "simplified_int643double3d",
      [](mesh_wrapper<int64_t, double, 3, 3> &wrapper, double error_rel,
         int optimize_iterations, double min_quality, bool preserve_boundary,
         double stabilizer, bool parallel, double feature_angle,
         double feature_weight, int iterations, int relaxation_iters,
         double lambda,
         std::optional<ndarray<numpy, const int, shape<-1>>> regions) {
        return simplified_impl<int64_t, double>(
            wrapper, error_rel, optimize_iterations, min_quality,
            preserve_boundary, stabilizer, parallel, feature_angle,
            feature_weight, iterations, relaxation_iters, lambda, regions);
      },
      arg("mesh"), arg("error_rel") = 0.002, arg("optimize_iterations") = 3,
      arg("min_quality") = 0.3, arg("preserve_boundary") = true,
      arg("stabilizer") = 1e-3, arg("parallel") = true,
      arg("feature_angle") = -1.0, arg("feature_weight") = 100.0,
      arg("iterations") = 1, arg("relaxation_iters") = 3, arg("lambda") = 0.5,
      arg("preserve_regions") = nanobind::none());
}

} // namespace tf::py

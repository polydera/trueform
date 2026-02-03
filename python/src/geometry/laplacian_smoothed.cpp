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

#include "trueform/python/geometry/laplacian_smoothed.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_laplacian_smoothed(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // ==========================================================================
  // LAPLACIAN_SMOOTHED
  // Takes points, vertex_link, iterations, lambda
  // Index types: int32, int64
  // Real types: float32, float64
  // Dims: 3
  // Suffix pattern: {index}_{real}_{dims} (e.g., int_float_3)
  // Total: 4 bindings
  // ==========================================================================

  // int32, float32, 3D
  m.def(
      "laplacian_smoothed_int_float_3",
      [](ndarray<numpy, float, shape<-1, 3>> points,
         const offset_blocked_array_wrapper<int, int> &vertex_link,
         std::size_t iterations, float lambda) {
        return laplacian_smoothed<int, float, 3>(points, vertex_link,
                                                  iterations, lambda);
      },
      arg("points"), arg("vertex_link"), arg("iterations"),
      arg("lambda_") = 0.5f);

  // int32, float64, 3D
  m.def(
      "laplacian_smoothed_int_double_3",
      [](ndarray<numpy, double, shape<-1, 3>> points,
         const offset_blocked_array_wrapper<int, int> &vertex_link,
         std::size_t iterations, double lambda) {
        return laplacian_smoothed<int, double, 3>(points, vertex_link,
                                                   iterations, lambda);
      },
      arg("points"), arg("vertex_link"), arg("iterations"),
      arg("lambda_") = 0.5);

  // int64, float32, 3D
  m.def(
      "laplacian_smoothed_int64_float_3",
      [](ndarray<numpy, float, shape<-1, 3>> points,
         const offset_blocked_array_wrapper<int64_t, int64_t> &vertex_link,
         std::size_t iterations, float lambda) {
        return laplacian_smoothed<int64_t, float, 3>(points, vertex_link,
                                                      iterations, lambda);
      },
      arg("points"), arg("vertex_link"), arg("iterations"),
      arg("lambda_") = 0.5f);

  // int64, float64, 3D
  m.def(
      "laplacian_smoothed_int64_double_3",
      [](ndarray<numpy, double, shape<-1, 3>> points,
         const offset_blocked_array_wrapper<int64_t, int64_t> &vertex_link,
         std::size_t iterations, double lambda) {
        return laplacian_smoothed<int64_t, double, 3>(points, vertex_link,
                                                       iterations, lambda);
      },
      arg("points"), arg("vertex_link"), arg("iterations"),
      arg("lambda_") = 0.5);
}

} // namespace tf::py

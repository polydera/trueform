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

#include "trueform/python/topology/make_neighborhoods.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_topology_make_neighborhoods(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // ==========================================================================
  // MAKE_NEIGHBORHOODS
  // Takes connectivity (vertex link), points, and radius
  // Index types: int32, int64
  // Real types: float32, float64
  // Dims: 3
  // Total: 4 bindings
  // ==========================================================================

  // int32, float32, 3D
  m.def(
      "make_neighborhoods_int_float_3",
      [](const offset_blocked_array_wrapper<int, int> &connectivity,
         ndarray<numpy, float, shape<-1, 3>> points, float radius,
         bool inclusive) {
        return make_neighborhoods<int, float, 3>(connectivity, points, radius,
                                                 inclusive);
      },
      arg("connectivity"), arg("points"), arg("radius"),
      arg("inclusive") = false);

  // int32, float64, 3D
  m.def(
      "make_neighborhoods_int_double_3",
      [](const offset_blocked_array_wrapper<int, int> &connectivity,
         ndarray<numpy, double, shape<-1, 3>> points, double radius,
         bool inclusive) {
        return make_neighborhoods<int, double, 3>(connectivity, points, radius,
                                                  inclusive);
      },
      arg("connectivity"), arg("points"), arg("radius"),
      arg("inclusive") = false);

  // int64, float32, 3D
  m.def(
      "make_neighborhoods_int64_float_3",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &connectivity,
         ndarray<numpy, float, shape<-1, 3>> points, float radius,
         bool inclusive) {
        return make_neighborhoods<int64_t, float, 3>(connectivity, points,
                                                     radius, inclusive);
      },
      arg("connectivity"), arg("points"), arg("radius"),
      arg("inclusive") = false);

  // int64, float64, 3D
  m.def(
      "make_neighborhoods_int64_double_3",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &connectivity,
         ndarray<numpy, double, shape<-1, 3>> points, double radius,
         bool inclusive) {
        return make_neighborhoods<int64_t, double, 3>(connectivity, points,
                                                      radius, inclusive);
      },
      arg("connectivity"), arg("points"), arg("radius"),
      arg("inclusive") = false);
}

} // namespace tf::py

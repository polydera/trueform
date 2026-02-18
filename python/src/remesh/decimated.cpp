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

#include "trueform/python/remesh/decimated_impl.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

namespace tf::py {

auto register_decimated(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // int32, float32, 3D
  m.def(
      "decimated_int3float3d",
      [](mesh_wrapper<int, float, 3, 3> &wrapper, float target_proportion,
         float max_aspect_ratio, bool preserve_boundary, double stabilizer,
         bool parallel) {
        return decimated_impl<int, float>(wrapper, target_proportion,
                                          max_aspect_ratio, preserve_boundary,
                                          stabilizer, parallel);
      },
      arg("mesh"), arg("target_proportion"), arg("max_aspect_ratio") = 40.f,
      arg("preserve_boundary") = false, arg("stabilizer") = 1e-3,
      arg("parallel") = true);

  // int32, float64, 3D
  m.def(
      "decimated_int3double3d",
      [](mesh_wrapper<int, double, 3, 3> &wrapper, double target_proportion,
         double max_aspect_ratio, bool preserve_boundary, double stabilizer,
         bool parallel) {
        return decimated_impl<int, double>(wrapper, target_proportion,
                                           max_aspect_ratio, preserve_boundary,
                                           stabilizer, parallel);
      },
      arg("mesh"), arg("target_proportion"), arg("max_aspect_ratio") = 40.0,
      arg("preserve_boundary") = false, arg("stabilizer") = 1e-3,
      arg("parallel") = true);

  // int64, float32, 3D
  m.def(
      "decimated_int643float3d",
      [](mesh_wrapper<int64_t, float, 3, 3> &wrapper, float target_proportion,
         float max_aspect_ratio, bool preserve_boundary, double stabilizer,
         bool parallel) {
        return decimated_impl<int64_t, float>(
            wrapper, target_proportion, max_aspect_ratio, preserve_boundary,
            stabilizer, parallel);
      },
      arg("mesh"), arg("target_proportion"), arg("max_aspect_ratio") = 40.f,
      arg("preserve_boundary") = false, arg("stabilizer") = 1e-3,
      arg("parallel") = true);

  // int64, float64, 3D
  m.def(
      "decimated_int643double3d",
      [](mesh_wrapper<int64_t, double, 3, 3> &wrapper,
         double target_proportion, double max_aspect_ratio,
         bool preserve_boundary, double stabilizer, bool parallel) {
        return decimated_impl<int64_t, double>(
            wrapper, target_proportion, max_aspect_ratio, preserve_boundary,
            stabilizer, parallel);
      },
      arg("mesh"), arg("target_proportion"), arg("max_aspect_ratio") = 40.0,
      arg("preserve_boundary") = false, arg("stabilizer") = 1e-3,
      arg("parallel") = true);
}

} // namespace tf::py

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

#include "trueform/python/reindex/split_into_components_impl.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_reindex_split_into_components(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // ==========================================================================
  // SPLIT_INTO_COMPONENTS - V=2 (Edges), V=3 (Triangles), Dynamic
  // Index types: int32, int64
  // Real types: float, double
  // Dims: 2D, 3D
  // Total: 3 types × 2 Index × 2 Real × 2 Dims = 24 bindings
  // ==========================================================================

  // V=2 (Edges), Dims=2, int32, float32
  m.def(
      "split_into_components_int2float2d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 2, float, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=2 (Edges), Dims=2, int32, float64
  m.def(
      "split_into_components_int2double2d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 2, double, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=2 (Edges), Dims=2, int64, float32
  m.def(
      "split_into_components_int642float2d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 2, float, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=2 (Edges), Dims=2, int64, float64
  m.def(
      "split_into_components_int642double2d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 2, double, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=2 (Edges), Dims=3, int32, float32
  m.def(
      "split_into_components_int2float3d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 2, float, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=2 (Edges), Dims=3, int32, float64
  m.def(
      "split_into_components_int2double3d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 2, double, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=2 (Edges), Dims=3, int64, float32
  m.def(
      "split_into_components_int642float3d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 2, float, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=2 (Edges), Dims=3, int64, float64
  m.def(
      "split_into_components_int642double3d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 2, double, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=2, int32, float32
  m.def(
      "split_into_components_int3float2d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 3, float, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=2, int32, float64
  m.def(
      "split_into_components_int3double2d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 3, double, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=2, int64, float32
  m.def(
      "split_into_components_int643float2d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 3, float, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=2, int64, float64
  m.def(
      "split_into_components_int643double2d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 3, double, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=3, int32, float32
  m.def(
      "split_into_components_int3float3d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 3, float, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=3, int32, float64
  m.def(
      "split_into_components_int3double3d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 3, double, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=3, int64, float32
  m.def(
      "split_into_components_int643float3d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 3, float, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=3, int64, float64
  m.def(
      "split_into_components_int643double3d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 3, double, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // Dynamic, Dims=2, int32, float32
  m.def(
      "split_into_components_intdynfloat2d",
      [](const offset_blocked_array_wrapper<int, int> &indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl_dynamic<int, float, 2>(indices, points,
                                                                  labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // Dynamic, Dims=2, int32, float64
  m.def(
      "split_into_components_intdyndouble2d",
      [](const offset_blocked_array_wrapper<int, int> &indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl_dynamic<int, double, 2>(indices, points,
                                                                   labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // Dynamic, Dims=2, int64, float32
  m.def(
      "split_into_components_int64dynfloat2d",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl_dynamic<int64_t, float, 2>(indices,
                                                                      points,
                                                                      labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // Dynamic, Dims=2, int64, float64
  m.def(
      "split_into_components_int64dyndouble2d",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl_dynamic<int64_t, double, 2>(indices,
                                                                       points,
                                                                       labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // Dynamic, Dims=3, int32, float32
  m.def(
      "split_into_components_intdynfloat3d",
      [](const offset_blocked_array_wrapper<int, int> &indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl_dynamic<int, float, 3>(indices, points,
                                                                  labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // Dynamic, Dims=3, int32, float64
  m.def(
      "split_into_components_intdyndouble3d",
      [](const offset_blocked_array_wrapper<int, int> &indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl_dynamic<int, double, 3>(indices, points,
                                                                   labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // Dynamic, Dims=3, int64, float32
  m.def(
      "split_into_components_int64dynfloat3d",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl_dynamic<int64_t, float, 3>(indices,
                                                                      points,
                                                                      labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // Dynamic, Dims=3, int64, float64
  m.def(
      "split_into_components_int64dyndouble3d",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl_dynamic<int64_t, double, 3>(indices,
                                                                       points,
                                                                       labels);
      },
      arg("indices"), arg("points"), arg("labels"));
}

} // namespace tf::py

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/reindex/split_into_components_impl.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_split_into_components(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // ==========================================================================
  // SPLIT_INTO_COMPONENTS - V=2 (Edges), V=3 (Triangles), V=4 (Quads)
  // Index types: int32, int64
  // Real types: float, double
  // Dims: 2D, 3D
  // Total: 3 V × 2 Index × 2 Real × 2 Dims = 24 bindings
  // ==========================================================================

  // V=2 (Edges), Dims=2, int32, float32
  m.def(
      "split_into_components_2intfloat2d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 2, float, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=2 (Edges), Dims=2, int32, float64
  m.def(
      "split_into_components_2intdouble2d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 2, double, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=2 (Edges), Dims=2, int64, float32
  m.def(
      "split_into_components_2int64float2d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 2, float, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=2 (Edges), Dims=2, int64, float64
  m.def(
      "split_into_components_2int64double2d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 2, double, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=2 (Edges), Dims=3, int32, float32
  m.def(
      "split_into_components_2intfloat3d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 2, float, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=2 (Edges), Dims=3, int32, float64
  m.def(
      "split_into_components_2intdouble3d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 2, double, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=2 (Edges), Dims=3, int64, float32
  m.def(
      "split_into_components_2int64float3d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 2, float, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=2 (Edges), Dims=3, int64, float64
  m.def(
      "split_into_components_2int64double3d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 2, double, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=2, int32, float32
  m.def(
      "split_into_components_3intfloat2d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 3, float, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=2, int32, float64
  m.def(
      "split_into_components_3intdouble2d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 3, double, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=2, int64, float32
  m.def(
      "split_into_components_3int64float2d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 3, float, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=2, int64, float64
  m.def(
      "split_into_components_3int64double2d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 3, double, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=3, int32, float32
  m.def(
      "split_into_components_3intfloat3d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 3, float, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=3, int32, float64
  m.def(
      "split_into_components_3intdouble3d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 3, double, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=3, int64, float32
  m.def(
      "split_into_components_3int64float3d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 3, float, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=3 (Triangles), Dims=3, int64, float64
  m.def(
      "split_into_components_3int64double3d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 3, double, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=4 (Quads), Dims=2, int32, float32
  m.def(
      "split_into_components_4intfloat2d",
      [](ndarray<numpy, const int, shape<-1, 4>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 4, float, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=4 (Quads), Dims=2, int32, float64
  m.def(
      "split_into_components_4intdouble2d",
      [](ndarray<numpy, const int, shape<-1, 4>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 4, double, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=4 (Quads), Dims=2, int64, float32
  m.def(
      "split_into_components_4int64float2d",
      [](ndarray<numpy, const int64_t, shape<-1, 4>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 4, float, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=4 (Quads), Dims=2, int64, float64
  m.def(
      "split_into_components_4int64double2d",
      [](ndarray<numpy, const int64_t, shape<-1, 4>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 4, double, 2>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=4 (Quads), Dims=3, int32, float32
  m.def(
      "split_into_components_4intfloat3d",
      [](ndarray<numpy, const int, shape<-1, 4>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 4, float, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=4 (Quads), Dims=3, int32, float64
  m.def(
      "split_into_components_4intdouble3d",
      [](ndarray<numpy, const int, shape<-1, 4>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int, 4, double, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=4 (Quads), Dims=3, int64, float32
  m.def(
      "split_into_components_4int64float3d",
      [](ndarray<numpy, const int64_t, shape<-1, 4>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 4, float, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));

  // V=4 (Quads), Dims=3, int64, float64
  m.def(
      "split_into_components_4int64double3d",
      [](ndarray<numpy, const int64_t, shape<-1, 4>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> labels) {
        return split_into_components_impl<int64_t, 4, double, 3>(indices, points, labels);
      },
      arg("indices"), arg("points"), arg("labels"));
}

} // namespace tf::py

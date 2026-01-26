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

#include "trueform/python/reindex/reindex_impl.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace tf::py {

auto register_reindex_reindex_by_ids(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // ==========================================================================
  // POINTS REINDEX_BY_IDS (8 bindings)
  // ==========================================================================

  // 2D Points - int32, float32
  m.def(
      "reindexed_by_ids_points_intfloat2d",
      [](ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl<int, float, 2>(points, ids);
      },
      arg("points"), arg("ids"));

  // 2D Points - int32, float64
  m.def(
      "reindexed_by_ids_points_intdouble2d",
      [](ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl<int, double, 2>(points, ids);
      },
      arg("points"), arg("ids"));

  // 2D Points - int64, float32
  m.def(
      "reindexed_by_ids_points_int64float2d",
      [](ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl<int64_t, float, 2>(points, ids);
      },
      arg("points"), arg("ids"));

  // 2D Points - int64, float64
  m.def(
      "reindexed_by_ids_points_int64double2d",
      [](ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl<int64_t, double, 2>(points, ids);
      },
      arg("points"), arg("ids"));

  // 3D Points - int32, float32
  m.def(
      "reindexed_by_ids_points_intfloat3d",
      [](ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl<int, float, 3>(points, ids);
      },
      arg("points"), arg("ids"));

  // 3D Points - int32, float64
  m.def(
      "reindexed_by_ids_points_intdouble3d",
      [](ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl<int, double, 3>(points, ids);
      },
      arg("points"), arg("ids"));

  // 3D Points - int64, float32
  m.def(
      "reindexed_by_ids_points_int64float3d",
      [](ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl<int64_t, float, 3>(points, ids);
      },
      arg("points"), arg("ids"));

  // 3D Points - int64, float64
  m.def(
      "reindexed_by_ids_points_int64double3d",
      [](ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl<int64_t, double, 3>(points, ids);
      },
      arg("points"), arg("ids"));

  // ==========================================================================
  // INDEXED GEOMETRY REINDEX_BY_IDS (24 bindings)
  // ==========================================================================

  // V=2 (Edges), Dims=2, int32, float32
  m.def(
      "reindexed_by_ids_indexed_int2float2d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl<int, 2, float, 2>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=2 (Edges), Dims=2, int32, float64
  m.def(
      "reindexed_by_ids_indexed_int2double2d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl<int, 2, double, 2>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=2 (Edges), Dims=2, int64, float32
  m.def(
      "reindexed_by_ids_indexed_int642float2d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl<int64_t, 2, float, 2>(indices, points,
                                                            ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=2 (Edges), Dims=2, int64, float64
  m.def(
      "reindexed_by_ids_indexed_int642double2d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl<int64_t, 2, double, 2>(indices, points,
                                                             ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=2 (Edges), Dims=3, int32, float32
  m.def(
      "reindexed_by_ids_indexed_int2float3d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl<int, 2, float, 3>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=2 (Edges), Dims=3, int32, float64
  m.def(
      "reindexed_by_ids_indexed_int2double3d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl<int, 2, double, 3>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=2 (Edges), Dims=3, int64, float32
  m.def(
      "reindexed_by_ids_indexed_int642float3d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl<int64_t, 2, float, 3>(indices, points,
                                                            ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=2 (Edges), Dims=3, int64, float64
  m.def(
      "reindexed_by_ids_indexed_int642double3d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl<int64_t, 2, double, 3>(indices, points,
                                                             ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=3 (Triangles), Dims=2, int32, float32
  m.def(
      "reindexed_by_ids_indexed_int3float2d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl<int, 3, float, 2>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=3 (Triangles), Dims=2, int32, float64
  m.def(
      "reindexed_by_ids_indexed_int3double2d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl<int, 3, double, 2>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=3 (Triangles), Dims=2, int64, float32
  m.def(
      "reindexed_by_ids_indexed_int643float2d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl<int64_t, 3, float, 2>(indices, points,
                                                            ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=3 (Triangles), Dims=2, int64, float64
  m.def(
      "reindexed_by_ids_indexed_int643double2d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl<int64_t, 3, double, 2>(indices, points,
                                                             ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=3 (Triangles), Dims=3, int32, float32
  m.def(
      "reindexed_by_ids_indexed_int3float3d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl<int, 3, float, 3>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=3 (Triangles), Dims=3, int32, float64
  m.def(
      "reindexed_by_ids_indexed_int3double3d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl<int, 3, double, 3>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=3 (Triangles), Dims=3, int64, float32
  m.def(
      "reindexed_by_ids_indexed_int643float3d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl<int64_t, 3, float, 3>(indices, points,
                                                            ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // V=3 (Triangles), Dims=3, int64, float64
  m.def(
      "reindexed_by_ids_indexed_int643double3d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl<int64_t, 3, double, 3>(indices, points,
                                                             ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // ==========================================================================
  // DYNAMIC INDEXED GEOMETRY REINDEX_BY_IDS (8 bindings)
  // ==========================================================================

  // Dynamic, Dims=2, int32, float32
  m.def(
      "reindexed_by_ids_indexed_intdynfloat2d",
      [](const offset_blocked_array_wrapper<int, int> &indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl_dynamic<int, float, 2>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // Dynamic, Dims=2, int32, float64
  m.def(
      "reindexed_by_ids_indexed_intdyndouble2d",
      [](const offset_blocked_array_wrapper<int, int> &indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl_dynamic<int, double, 2>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // Dynamic, Dims=2, int64, float32
  m.def(
      "reindexed_by_ids_indexed_int64dynfloat2d",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl_dynamic<int64_t, float, 2>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // Dynamic, Dims=2, int64, float64
  m.def(
      "reindexed_by_ids_indexed_int64dyndouble2d",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl_dynamic<int64_t, double, 2>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // Dynamic, Dims=3, int32, float32
  m.def(
      "reindexed_by_ids_indexed_intdynfloat3d",
      [](const offset_blocked_array_wrapper<int, int> &indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl_dynamic<int, float, 3>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // Dynamic, Dims=3, int32, float64
  m.def(
      "reindexed_by_ids_indexed_intdyndouble3d",
      [](const offset_blocked_array_wrapper<int, int> &indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int, shape<-1>> ids) {
        return reindexed_by_ids_impl_dynamic<int, double, 3>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // Dynamic, Dims=3, int64, float32
  m.def(
      "reindexed_by_ids_indexed_int64dynfloat3d",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl_dynamic<int64_t, float, 3>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));

  // Dynamic, Dims=3, int64, float64
  m.def(
      "reindexed_by_ids_indexed_int64dyndouble3d",
      [](const offset_blocked_array_wrapper<int64_t, int64_t> &indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         ndarray<numpy, const int64_t, shape<-1>> ids) {
        return reindexed_by_ids_impl_dynamic<int64_t, double, 3>(indices, points, ids);
      },
      arg("indices"), arg("points"), arg("ids"));
}

} // namespace tf::py

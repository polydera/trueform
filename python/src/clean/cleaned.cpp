/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/clean/cleaned_impl.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

namespace tf::py {

auto register_clean_cleaned(nanobind::module_ &m) -> void {
  using namespace nanobind;

  // ==========================================================================
  // POINTS CLEANING (4 bindings - always with maps)
  // ==========================================================================

  // 2D Points - float32
  m.def(
      "cleaned_points_with_maps_float2d",
      [](ndarray<numpy, const float, shape<-1, 2>> points,
         std::optional<float> tolerance) {
        return cleaned_impl<int64_t, float, 2>(points, tolerance,
                                               tf::return_index_map);
      },
      arg("points"), arg("tolerance").none() = nanobind::none());

  // 2D Points - float64
  m.def(
      "cleaned_points_with_maps_double2d",
      [](ndarray<numpy, const double, shape<-1, 2>> points,
         std::optional<double> tolerance) {
        return cleaned_impl<int64_t, double, 2>(points, tolerance,
                                                tf::return_index_map);
      },
      arg("points"), arg("tolerance").none() = nanobind::none());

  // 3D Points - float32
  m.def(
      "cleaned_points_with_maps_float3d",
      [](ndarray<numpy, const float, shape<-1, 3>> points,
         std::optional<float> tolerance) {
        return cleaned_impl<int64_t, float, 3>(points, tolerance,
                                               tf::return_index_map);
      },
      arg("points"), arg("tolerance").none() = nanobind::none());

  // 3D Points - float64
  m.def(
      "cleaned_points_with_maps_double3d",
      [](ndarray<numpy, const double, shape<-1, 3>> points,
         std::optional<double> tolerance) {
        return cleaned_impl<int64_t, double, 3>(points, tolerance,
                                                tf::return_index_map);
      },
      arg("points"), arg("tolerance").none() = nanobind::none());

  // ==========================================================================
  // SOUP CLEANING (12 bindings - soups never return maps)
  // ==========================================================================

  // V=2 (Segments), Dims=2, float32
  m.def(
      "cleaned_soup_2float2d",
      [](ndarray<numpy, const float, shape<-1, 2, 2>> soup,
         std::optional<float> tolerance) {
        return cleaned_impl<int, float, 2, 2>(soup, tolerance);
      },
      arg("soup"), arg("tolerance").none() = nanobind::none());

  // V=2 (Segments), Dims=2, float64
  m.def(
      "cleaned_soup_2double2d",
      [](ndarray<numpy, const double, shape<-1, 2, 2>> soup,
         std::optional<double> tolerance) {
        return cleaned_impl<int, double, 2, 2>(soup, tolerance);
      },
      arg("soup"), arg("tolerance").none() = nanobind::none());

  // V=2 (Segments), Dims=3, float32
  m.def(
      "cleaned_soup_2float3d",
      [](ndarray<numpy, const float, shape<-1, 2, 3>> soup,
         std::optional<float> tolerance) {
        return cleaned_impl<int, float, 2, 3>(soup, tolerance);
      },
      arg("soup"), arg("tolerance").none() = nanobind::none());

  // V=2 (Segments), Dims=3, float64
  m.def(
      "cleaned_soup_2double3d",
      [](ndarray<numpy, const double, shape<-1, 2, 3>> soup,
         std::optional<double> tolerance) {
        return cleaned_impl<int, double, 2, 3>(soup, tolerance);
      },
      arg("soup"), arg("tolerance").none() = nanobind::none());

  // V=3 (Triangles), Dims=2, float32
  m.def(
      "cleaned_soup_3float2d",
      [](ndarray<numpy, const float, shape<-1, 3, 2>> soup,
         std::optional<float> tolerance) {
        return cleaned_impl<int, float, 3, 2>(soup, tolerance);
      },
      arg("soup"), arg("tolerance").none() = nanobind::none());

  // V=3 (Triangles), Dims=2, float64
  m.def(
      "cleaned_soup_3double2d",
      [](ndarray<numpy, const double, shape<-1, 3, 2>> soup,
         std::optional<double> tolerance) {
        return cleaned_impl<int, double, 3, 2>(soup, tolerance);
      },
      arg("soup"), arg("tolerance").none() = nanobind::none());

  // V=3 (Triangles), Dims=3, float32
  m.def(
      "cleaned_soup_3float3d",
      [](ndarray<numpy, const float, shape<-1, 3, 3>> soup,
         std::optional<float> tolerance) {
        return cleaned_impl<int, float, 3, 3>(soup, tolerance);
      },
      arg("soup"), arg("tolerance").none() = nanobind::none());

  // V=3 (Triangles), Dims=3, float64
  m.def(
      "cleaned_soup_3double3d",
      [](ndarray<numpy, const double, shape<-1, 3, 3>> soup,
         std::optional<double> tolerance) {
        return cleaned_impl<int, double, 3, 3>(soup, tolerance);
      },
      arg("soup"), arg("tolerance").none() = nanobind::none());

  // V=4 (Quads), Dims=2, float32
  m.def(
      "cleaned_soup_4float2d",
      [](ndarray<numpy, const float, shape<-1, 4, 2>> soup,
         std::optional<float> tolerance) {
        return cleaned_impl<int, float, 4, 2>(soup, tolerance);
      },
      arg("soup"), arg("tolerance").none() = nanobind::none());

  // V=4 (Quads), Dims=2, float64
  m.def(
      "cleaned_soup_4double2d",
      [](ndarray<numpy, const double, shape<-1, 4, 2>> soup,
         std::optional<double> tolerance) {
        return cleaned_impl<int, double, 4, 2>(soup, tolerance);
      },
      arg("soup"), arg("tolerance").none() = nanobind::none());

  // V=4 (Quads), Dims=3, float32
  m.def(
      "cleaned_soup_4float3d",
      [](ndarray<numpy, const float, shape<-1, 4, 3>> soup,
         std::optional<float> tolerance) {
        return cleaned_impl<int, float, 4, 3>(soup, tolerance);
      },
      arg("soup"), arg("tolerance").none() = nanobind::none());

  // V=4 (Quads), Dims=3, float64
  m.def(
      "cleaned_soup_4double3d",
      [](ndarray<numpy, const double, shape<-1, 4, 3>> soup,
         std::optional<double> tolerance) {
        return cleaned_impl<int, double, 4, 3>(soup, tolerance);
      },
      arg("soup"), arg("tolerance").none() = nanobind::none());

  // ==========================================================================
  // INDEXED GEOMETRY CLEANING (24 bindings - always with maps)
  // ==========================================================================

  // V=2 (Edges), Dims=2, int32, float32
  m.def(
      "cleaned_indexed_with_maps_2intfloat2d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         std::optional<float> tolerance) {
        return cleaned_impl<int, 2, float, 2>(indices, points, tolerance,
                                              tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=2 (Edges), Dims=2, int32, float64
  m.def(
      "cleaned_indexed_with_maps_2intdouble2d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         std::optional<double> tolerance) {
        return cleaned_impl<int, 2, double, 2>(indices, points, tolerance,
                                               tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=2 (Edges), Dims=2, int64, float32
  m.def(
      "cleaned_indexed_with_maps_2int64float2d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         std::optional<float> tolerance) {
        return cleaned_impl<int64_t, 2, float, 2>(indices, points, tolerance,
                                                   tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=2 (Edges), Dims=2, int64, float64
  m.def(
      "cleaned_indexed_with_maps_2int64double2d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         std::optional<double> tolerance) {
        return cleaned_impl<int64_t, 2, double, 2>(indices, points, tolerance,
                                                    tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=2 (Edges), Dims=3, int32, float32
  m.def(
      "cleaned_indexed_with_maps_2intfloat3d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         std::optional<float> tolerance) {
        return cleaned_impl<int, 2, float, 3>(indices, points, tolerance,
                                              tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=2 (Edges), Dims=3, int32, float64
  m.def(
      "cleaned_indexed_with_maps_2intdouble3d",
      [](ndarray<numpy, const int, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         std::optional<double> tolerance) {
        return cleaned_impl<int, 2, double, 3>(indices, points, tolerance,
                                               tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=2 (Edges), Dims=3, int64, float32
  m.def(
      "cleaned_indexed_with_maps_2int64float3d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         std::optional<float> tolerance) {
        return cleaned_impl<int64_t, 2, float, 3>(indices, points, tolerance,
                                                   tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=2 (Edges), Dims=3, int64, float64
  m.def(
      "cleaned_indexed_with_maps_2int64double3d",
      [](ndarray<numpy, const int64_t, shape<-1, 2>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         std::optional<double> tolerance) {
        return cleaned_impl<int64_t, 2, double, 3>(indices, points, tolerance,
                                                    tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=3 (Triangles), Dims=2, int32, float32
  m.def(
      "cleaned_indexed_with_maps_3intfloat2d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         std::optional<float> tolerance) {
        return cleaned_impl<int, 3, float, 2>(indices, points, tolerance,
                                              tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=3 (Triangles), Dims=2, int32, float64
  m.def(
      "cleaned_indexed_with_maps_3intdouble2d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         std::optional<double> tolerance) {
        return cleaned_impl<int, 3, double, 2>(indices, points, tolerance,
                                               tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=3 (Triangles), Dims=2, int64, float32
  m.def(
      "cleaned_indexed_with_maps_3int64float2d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         std::optional<float> tolerance) {
        return cleaned_impl<int64_t, 3, float, 2>(indices, points, tolerance,
                                                   tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=3 (Triangles), Dims=2, int64, float64
  m.def(
      "cleaned_indexed_with_maps_3int64double2d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         std::optional<double> tolerance) {
        return cleaned_impl<int64_t, 3, double, 2>(indices, points, tolerance,
                                                    tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=3 (Triangles), Dims=3, int32, float32
  m.def(
      "cleaned_indexed_with_maps_3intfloat3d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         std::optional<float> tolerance) {
        return cleaned_impl<int, 3, float, 3>(indices, points, tolerance,
                                              tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=3 (Triangles), Dims=3, int32, float64
  m.def(
      "cleaned_indexed_with_maps_3intdouble3d",
      [](ndarray<numpy, const int, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         std::optional<double> tolerance) {
        return cleaned_impl<int, 3, double, 3>(indices, points, tolerance,
                                               tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=3 (Triangles), Dims=3, int64, float32
  m.def(
      "cleaned_indexed_with_maps_3int64float3d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         std::optional<float> tolerance) {
        return cleaned_impl<int64_t, 3, float, 3>(indices, points, tolerance,
                                                   tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=3 (Triangles), Dims=3, int64, float64
  m.def(
      "cleaned_indexed_with_maps_3int64double3d",
      [](ndarray<numpy, const int64_t, shape<-1, 3>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         std::optional<double> tolerance) {
        return cleaned_impl<int64_t, 3, double, 3>(indices, points, tolerance,
                                                    tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=4 (Quads), Dims=2, int32, float32
  m.def(
      "cleaned_indexed_with_maps_4intfloat2d",
      [](ndarray<numpy, const int, shape<-1, 4>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         std::optional<float> tolerance) {
        return cleaned_impl<int, 4, float, 2>(indices, points, tolerance,
                                              tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=4 (Quads), Dims=2, int32, float64
  m.def(
      "cleaned_indexed_with_maps_4intdouble2d",
      [](ndarray<numpy, const int, shape<-1, 4>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         std::optional<double> tolerance) {
        return cleaned_impl<int, 4, double, 2>(indices, points, tolerance,
                                               tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=4 (Quads), Dims=2, int64, float32
  m.def(
      "cleaned_indexed_with_maps_4int64float2d",
      [](ndarray<numpy, const int64_t, shape<-1, 4>> indices,
         ndarray<numpy, const float, shape<-1, 2>> points,
         std::optional<float> tolerance) {
        return cleaned_impl<int64_t, 4, float, 2>(indices, points, tolerance,
                                                   tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=4 (Quads), Dims=2, int64, float64
  m.def(
      "cleaned_indexed_with_maps_4int64double2d",
      [](ndarray<numpy, const int64_t, shape<-1, 4>> indices,
         ndarray<numpy, const double, shape<-1, 2>> points,
         std::optional<double> tolerance) {
        return cleaned_impl<int64_t, 4, double, 2>(indices, points, tolerance,
                                                    tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=4 (Quads), Dims=3, int32, float32
  m.def(
      "cleaned_indexed_with_maps_4intfloat3d",
      [](ndarray<numpy, const int, shape<-1, 4>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         std::optional<float> tolerance) {
        return cleaned_impl<int, 4, float, 3>(indices, points, tolerance,
                                              tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=4 (Quads), Dims=3, int32, float64
  m.def(
      "cleaned_indexed_with_maps_4intdouble3d",
      [](ndarray<numpy, const int, shape<-1, 4>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         std::optional<double> tolerance) {
        return cleaned_impl<int, 4, double, 3>(indices, points, tolerance,
                                               tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=4 (Quads), Dims=3, int64, float32
  m.def(
      "cleaned_indexed_with_maps_4int64float3d",
      [](ndarray<numpy, const int64_t, shape<-1, 4>> indices,
         ndarray<numpy, const float, shape<-1, 3>> points,
         std::optional<float> tolerance) {
        return cleaned_impl<int64_t, 4, float, 3>(indices, points, tolerance,
                                                   tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());

  // V=4 (Quads), Dims=3, int64, float64
  m.def(
      "cleaned_indexed_with_maps_4int64double3d",
      [](ndarray<numpy, const int64_t, shape<-1, 4>> indices,
         ndarray<numpy, const double, shape<-1, 3>> points,
         std::optional<double> tolerance) {
        return cleaned_impl<int64_t, 4, double, 3>(indices, points, tolerance,
                                                    tf::return_index_map);
      },
      arg("indices"), arg("points"), arg("tolerance").none() = nanobind::none());
}

} // namespace tf::py

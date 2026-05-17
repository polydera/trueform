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
#include <nanobind/ndarray.h>

#include <trueform/core/curve.hpp>
#include <trueform/core/curves.hpp>
#include <trueform/core/points.hpp>
#include <trueform/geometry/make_tube_mesh.hpp>
#include <trueform/python/core/offset_blocked_array.hpp>
#include <trueform/python/util/make_numpy_array.hpp>

namespace nb = nanobind;

namespace tf::py {
namespace impl {

// ============================================================================
// Single curve (points only)
// ============================================================================

template <typename RealT, std::size_t Dims>
auto make_tube_mesh_single(
    nb::ndarray<nb::numpy, const RealT, nb::shape<-1, Dims>> points_arr,
    RealT radius, int radial_segments) {
  const RealT *points_data = points_arr.data();
  std::size_t num_points = points_arr.shape(0);

  auto points_range =
      tf::make_points<Dims>(tf::make_range(points_data, num_points * Dims));
  auto curve = tf::make_curve<Dims>(points_range);

  auto result = tf::make_tube_mesh<int>(curve, radius,
                                        static_cast<int>(radial_segments));
  auto [faces, points] = make_numpy_array(std::move(result));
  return nb::make_tuple(faces, points);
}

// ============================================================================
// Many curves (OffsetBlockedArray + points)
// ============================================================================

template <typename Index, typename RealT, std::size_t Dims>
auto make_tube_mesh_multi(
    const offset_blocked_array_wrapper<Index, Index> &paths_wrapper,
    nb::ndarray<nb::numpy, const RealT, nb::shape<-1, Dims>> points_arr,
    RealT radius, int radial_segments) {
  const RealT *points_data = points_arr.data();
  std::size_t num_points = points_arr.shape(0);

  auto paths_range = paths_wrapper.make_range();
  auto points_range =
      tf::make_points<Dims>(tf::make_range(points_data, num_points * Dims));
  auto curves = tf::make_curves(paths_range, points_range);

  auto result = tf::make_tube_mesh<Index>(
      curves, radius, static_cast<Index>(radial_segments));
  auto [faces, points] = make_numpy_array(std::move(result));
  return nb::make_tuple(faces, points);
}

} // namespace impl

// ============================================================================
// Registration
// ============================================================================

auto register_make_tube_mesh(nb::module_ &m) -> void {
  // Single-curve overloads (just points): 2 variants
  m.def("make_tube_mesh_float3d", &impl::make_tube_mesh_single<float, 3>,
        nb::arg("points"), nb::arg("radius"), nb::arg("radial_segments"),
        "Tube mesh around a single 3D polyline (float32 points).");
  m.def("make_tube_mesh_double3d", &impl::make_tube_mesh_single<double, 3>,
        nb::arg("points"), nb::arg("radius"), nb::arg("radial_segments"),
        "Tube mesh around a single 3D polyline (float64 points).");

  // Multi-curve overloads (paths-OBA + points): 4 variants
  m.def("make_tube_mesh_intdynfloat3d",
        &impl::make_tube_mesh_multi<int, float, 3>,
        nb::arg("paths"), nb::arg("points"), nb::arg("radius"),
        nb::arg("radial_segments"),
        "Tube mesh around many 3D polylines (int32 paths, float32 points).");
  m.def("make_tube_mesh_intdyndouble3d",
        &impl::make_tube_mesh_multi<int, double, 3>,
        nb::arg("paths"), nb::arg("points"), nb::arg("radius"),
        nb::arg("radial_segments"),
        "Tube mesh around many 3D polylines (int32 paths, float64 points).");
  m.def("make_tube_mesh_int64dynfloat3d",
        &impl::make_tube_mesh_multi<std::int64_t, float, 3>,
        nb::arg("paths"), nb::arg("points"), nb::arg("radius"),
        nb::arg("radial_segments"),
        "Tube mesh around many 3D polylines (int64 paths, float32 points).");
  m.def("make_tube_mesh_int64dyndouble3d",
        &impl::make_tube_mesh_multi<std::int64_t, double, 3>,
        nb::arg("paths"), nb::arg("points"), nb::arg("radius"),
        nb::arg("radial_segments"),
        "Tube mesh around many 3D polylines (int64 paths, float64 points).");
}

} // namespace tf::py

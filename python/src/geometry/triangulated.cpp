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
#include <trueform/geometry/triangulated.hpp>
#include <trueform/python/core/offset_blocked_array.hpp>
#include <trueform/python/util/make_numpy_array.hpp>

namespace nb = nanobind;

namespace tf::py {
namespace impl {

// ============================================================================
// Single polygon (points only)
// ============================================================================

template <typename RealT, std::size_t Dims>
auto triangulated_polygon(
    nb::ndarray<nb::numpy, const RealT, nb::shape<-1, Dims>> points_arr) {
  const RealT *points_data = points_arr.data();
  std::size_t num_points = points_arr.shape(0);

  auto points_range =
      tf::make_points<Dims>(tf::make_range(points_data, num_points * Dims));
  auto polygon = tf::make_polygon(points_range);

  auto result = tf::triangulated(polygon);
  auto [faces, points] = make_numpy_array(std::move(result));
  return nb::make_tuple(faces, points);
}

// ============================================================================
// Dynamic ngon mesh (OffsetBlockedArray)
// ============================================================================

template <typename Index, typename RealT, std::size_t Dims>
auto triangulated_dynamic(
    const offset_blocked_array_wrapper<Index, Index> &faces_wrapper,
    nb::ndarray<nb::numpy, const RealT, nb::shape<-1, Dims>> points_arr) {
  const RealT *points_data = points_arr.data();
  std::size_t num_points = points_arr.shape(0);

  auto faces_range = faces_wrapper.make_range();
  auto points_range =
      tf::make_points<Dims>(tf::make_range(points_data, num_points * Dims));
  auto polygons = tf::make_polygons(faces_range, points_range);

  auto result = tf::triangulated(polygons);
  auto [faces, points] = make_numpy_array(std::move(result));
  return nb::make_tuple(faces, points);
}

} // namespace impl

// ============================================================================
// Registration
// ============================================================================

auto register_triangulated(nb::module_ &m) -> void {

  // ========== Single polygon (points only) - 4 variants ==========

  m.def("triangulated_float2d", &impl::triangulated_polygon<float, 2>,
        nb::arg("points"), "Triangulate a single 2D polygon.");

  m.def("triangulated_float3d", &impl::triangulated_polygon<float, 3>,
        nb::arg("points"), "Triangulate a single 3D polygon.");

  m.def("triangulated_double2d", &impl::triangulated_polygon<double, 2>,
        nb::arg("points"), "Triangulate a single 2D polygon.");

  m.def("triangulated_double3d", &impl::triangulated_polygon<double, 3>,
        nb::arg("points"), "Triangulate a single 3D polygon.");

  // ========== Dynamic mesh (OffsetBlockedArray) - 8 variants ==========

  m.def("triangulated_intdynfloat2d", &impl::triangulated_dynamic<int, float, 2>,
        nb::arg("faces"), nb::arg("points"),
        "Triangulate a 2D dynamic polygon mesh.");

  m.def("triangulated_intdynfloat3d", &impl::triangulated_dynamic<int, float, 3>,
        nb::arg("faces"), nb::arg("points"),
        "Triangulate a 3D dynamic polygon mesh.");

  m.def("triangulated_intdyndouble2d",
        &impl::triangulated_dynamic<int, double, 2>, nb::arg("faces"),
        nb::arg("points"), "Triangulate a 2D dynamic polygon mesh.");

  m.def("triangulated_intdyndouble3d",
        &impl::triangulated_dynamic<int, double, 3>, nb::arg("faces"),
        nb::arg("points"), "Triangulate a 3D dynamic polygon mesh.");

  m.def("triangulated_int64dynfloat2d",
        &impl::triangulated_dynamic<int64_t, float, 2>, nb::arg("faces"),
        nb::arg("points"), "Triangulate a 2D dynamic polygon mesh.");

  m.def("triangulated_int64dynfloat3d",
        &impl::triangulated_dynamic<int64_t, float, 3>, nb::arg("faces"),
        nb::arg("points"), "Triangulate a 3D dynamic polygon mesh.");

  m.def("triangulated_int64dyndouble2d",
        &impl::triangulated_dynamic<int64_t, double, 2>, nb::arg("faces"),
        nb::arg("points"), "Triangulate a 2D dynamic polygon mesh.");

  m.def("triangulated_int64dyndouble3d",
        &impl::triangulated_dynamic<int64_t, double, 3>, nb::arg("faces"),
        nb::arg("points"), "Triangulate a 3D dynamic polygon mesh.");
}

} // namespace tf::py

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
#include <trueform/core/area.hpp>
#include <trueform/core/signed_volume.hpp>
#include <trueform/python/spatial/mesh.hpp>

namespace nb = nanobind;

namespace tf::py {
namespace impl {

// ============================================================================
// Signed volume (mesh only, 3D)
// ============================================================================

template <typename Index, std::size_t Ngon, typename RealT>
auto signed_volume_impl(const mesh_wrapper<Index, RealT, Ngon, 3> &mesh) {
  return tf::signed_volume(mesh.make_primitive_range());
}

// ============================================================================
// Area - single polygon (points only)
// ============================================================================

template <typename RealT, std::size_t Dims>
auto area_polygon(
    nb::ndarray<nb::numpy, const RealT, nb::shape<-1, Dims>> points_arr) {
  const RealT *points_data = points_arr.data();
  std::size_t num_points = points_arr.shape(0);

  auto points_range =
      tf::make_points<Dims>(tf::make_range(points_data, num_points * Dims));
  auto polygon = tf::make_polygon(points_range);

  return tf::area(polygon);
}

// ============================================================================
// Area - mesh (total surface area)
// ============================================================================

template <typename Index, std::size_t Ngon, std::size_t Dims, typename RealT>
auto area_mesh(const mesh_wrapper<Index, RealT, Ngon, Dims> &mesh) {
  return tf::area(mesh.make_primitive_range());
}

} // namespace impl

// ============================================================================
// Registration
// ============================================================================

auto register_measurements(nb::module_ &m) -> void {

  // ========== Signed volume - 3D only ==========

  // int32, ngon=3
  m.def("signed_volume_int3float3d", &impl::signed_volume_impl<int, 3, float>,
        nb::arg("mesh"), "Compute signed volume of a 3D mesh.");
  m.def("signed_volume_int3double3d", &impl::signed_volume_impl<int, 3, double>,
        nb::arg("mesh"), "Compute signed volume of a 3D mesh.");

  // int32, dynamic
  m.def("signed_volume_intdynfloat3d",
        &impl::signed_volume_impl<int, dynamic_size, float>, nb::arg("mesh"),
        "Compute signed volume of a 3D dynamic mesh.");
  m.def("signed_volume_intdyndouble3d",
        &impl::signed_volume_impl<int, dynamic_size, double>, nb::arg("mesh"),
        "Compute signed volume of a 3D dynamic mesh.");

  // int64, ngon=3
  m.def("signed_volume_int643float3d",
        &impl::signed_volume_impl<int64_t, 3, float>, nb::arg("mesh"),
        "Compute signed volume of a 3D mesh.");
  m.def("signed_volume_int643double3d",
        &impl::signed_volume_impl<int64_t, 3, double>, nb::arg("mesh"),
        "Compute signed volume of a 3D mesh.");

  // int64, dynamic
  m.def("signed_volume_int64dynfloat3d",
        &impl::signed_volume_impl<int64_t, dynamic_size, float>, nb::arg("mesh"),
        "Compute signed volume of a 3D dynamic mesh.");
  m.def("signed_volume_int64dyndouble3d",
        &impl::signed_volume_impl<int64_t, dynamic_size, double>, nb::arg("mesh"),
        "Compute signed volume of a 3D dynamic mesh.");

  // ========== Area - single polygon ==========

  m.def("area_float2d", &impl::area_polygon<float, 2>, nb::arg("points"),
        "Compute area of a 2D polygon.");
  m.def("area_float3d", &impl::area_polygon<float, 3>, nb::arg("points"),
        "Compute area of a 3D polygon.");
  m.def("area_double2d", &impl::area_polygon<double, 2>, nb::arg("points"),
        "Compute area of a 2D polygon.");
  m.def("area_double3d", &impl::area_polygon<double, 3>, nb::arg("points"),
        "Compute area of a 3D polygon.");

  // ========== Area - mesh, 2D ==========

  // int32, ngon=3
  m.def("area_int3float2d", &impl::area_mesh<int, 3, 2, float>, nb::arg("mesh"),
        "Compute total area of a 2D mesh.");
  m.def("area_int3double2d", &impl::area_mesh<int, 3, 2, double>,
        nb::arg("mesh"), "Compute total area of a 2D mesh.");

  // int32, dynamic
  m.def("area_intdynfloat2d", &impl::area_mesh<int, dynamic_size, 2, float>,
        nb::arg("mesh"), "Compute total area of a 2D dynamic mesh.");
  m.def("area_intdyndouble2d", &impl::area_mesh<int, dynamic_size, 2, double>,
        nb::arg("mesh"), "Compute total area of a 2D dynamic mesh.");

  // int64, ngon=3
  m.def("area_int643float2d", &impl::area_mesh<int64_t, 3, 2, float>,
        nb::arg("mesh"), "Compute total area of a 2D mesh.");
  m.def("area_int643double2d", &impl::area_mesh<int64_t, 3, 2, double>,
        nb::arg("mesh"), "Compute total area of a 2D mesh.");

  // int64, dynamic
  m.def("area_int64dynfloat2d", &impl::area_mesh<int64_t, dynamic_size, 2, float>,
        nb::arg("mesh"), "Compute total area of a 2D dynamic mesh.");
  m.def("area_int64dyndouble2d",
        &impl::area_mesh<int64_t, dynamic_size, 2, double>, nb::arg("mesh"),
        "Compute total area of a 2D dynamic mesh.");

  // ========== Area - mesh, 3D ==========

  // int32, ngon=3
  m.def("area_int3float3d", &impl::area_mesh<int, 3, 3, float>, nb::arg("mesh"),
        "Compute total area of a 3D mesh.");
  m.def("area_int3double3d", &impl::area_mesh<int, 3, 3, double>,
        nb::arg("mesh"), "Compute total area of a 3D mesh.");

  // int32, dynamic
  m.def("area_intdynfloat3d", &impl::area_mesh<int, dynamic_size, 3, float>,
        nb::arg("mesh"), "Compute total area of a 3D dynamic mesh.");
  m.def("area_intdyndouble3d", &impl::area_mesh<int, dynamic_size, 3, double>,
        nb::arg("mesh"), "Compute total area of a 3D dynamic mesh.");

  // int64, ngon=3
  m.def("area_int643float3d", &impl::area_mesh<int64_t, 3, 3, float>,
        nb::arg("mesh"), "Compute total area of a 3D mesh.");
  m.def("area_int643double3d", &impl::area_mesh<int64_t, 3, 3, double>,
        nb::arg("mesh"), "Compute total area of a 3D mesh.");

  // int64, dynamic
  m.def("area_int64dynfloat3d", &impl::area_mesh<int64_t, dynamic_size, 3, float>,
        nb::arg("mesh"), "Compute total area of a 3D dynamic mesh.");
  m.def("area_int64dyndouble3d",
        &impl::area_mesh<int64_t, dynamic_size, 3, double>, nb::arg("mesh"),
        "Compute total area of a 3D dynamic mesh.");
}

} // namespace tf::py

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
#include <trueform/geometry/make_sphere_mesh.hpp>
#include <trueform/geometry/make_cylinder_mesh.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/geometry/make_plane_mesh.hpp>
#include <trueform/python/util/make_numpy_array.hpp>

namespace nb = nanobind;

namespace tf::py {
namespace impl {

// ============================================================================
// Sphere mesh
// ============================================================================

template <typename Index, typename RealT>
auto make_sphere_mesh_impl(RealT radius, Index stacks, Index segments) {
  auto result = tf::make_sphere_mesh<Index>(radius, stacks, segments);
  auto [faces, points] = make_numpy_array(std::move(result));
  return nb::make_tuple(faces, points);
}

// ============================================================================
// Cylinder mesh
// ============================================================================

template <typename Index, typename RealT>
auto make_cylinder_mesh_impl(RealT radius, RealT height, Index segments) {
  auto result = tf::make_cylinder_mesh<Index>(radius, height, segments);
  auto [faces, points] = make_numpy_array(std::move(result));
  return nb::make_tuple(faces, points);
}

// ============================================================================
// Box mesh (simple)
// ============================================================================

template <typename Index, typename RealT>
auto make_box_mesh_impl(RealT width, RealT height, RealT depth) {
  auto result = tf::make_box_mesh<Index>(width, height, depth);
  auto [faces, points] = make_numpy_array(std::move(result));
  return nb::make_tuple(faces, points);
}

// ============================================================================
// Box mesh (subdivided)
// ============================================================================

template <typename Index, typename RealT>
auto make_box_mesh_subdivided_impl(RealT width, RealT height, RealT depth,
                                    Index width_ticks, Index height_ticks,
                                    Index depth_ticks) {
  auto result = tf::make_box_mesh<Index>(width, height, depth, width_ticks,
                                          height_ticks, depth_ticks);
  auto [faces, points] = make_numpy_array(std::move(result));
  return nb::make_tuple(faces, points);
}

// ============================================================================
// Plane mesh (subdivided - the base implementation)
// ============================================================================

template <typename Index, typename RealT>
auto make_plane_mesh_impl(RealT width, RealT height, Index width_ticks,
                          Index height_ticks) {
  auto result =
      tf::make_plane_mesh<Index>(width, height, width_ticks, height_ticks);
  auto [faces, points] = make_numpy_array(std::move(result));
  return nb::make_tuple(faces, points);
}

} // namespace impl

// ============================================================================
// Registration
// ============================================================================

auto register_make_mesh_primitives(nb::module_ &m) -> void {

  // ========== Sphere mesh - 4 variants ==========

  m.def("make_sphere_mesh_int_float",
        &impl::make_sphere_mesh_impl<int, float>,
        nb::arg("radius"), nb::arg("stacks"), nb::arg("segments"),
        "Create a UV sphere mesh.");

  m.def("make_sphere_mesh_int_double",
        &impl::make_sphere_mesh_impl<int, double>,
        nb::arg("radius"), nb::arg("stacks"), nb::arg("segments"),
        "Create a UV sphere mesh.");

  m.def("make_sphere_mesh_int64_float",
        &impl::make_sphere_mesh_impl<int64_t, float>,
        nb::arg("radius"), nb::arg("stacks"), nb::arg("segments"),
        "Create a UV sphere mesh.");

  m.def("make_sphere_mesh_int64_double",
        &impl::make_sphere_mesh_impl<int64_t, double>,
        nb::arg("radius"), nb::arg("stacks"), nb::arg("segments"),
        "Create a UV sphere mesh.");

  // ========== Cylinder mesh - 4 variants ==========

  m.def("make_cylinder_mesh_int_float",
        &impl::make_cylinder_mesh_impl<int, float>,
        nb::arg("radius"), nb::arg("height"), nb::arg("segments"),
        "Create a cylinder mesh.");

  m.def("make_cylinder_mesh_int_double",
        &impl::make_cylinder_mesh_impl<int, double>,
        nb::arg("radius"), nb::arg("height"), nb::arg("segments"),
        "Create a cylinder mesh.");

  m.def("make_cylinder_mesh_int64_float",
        &impl::make_cylinder_mesh_impl<int64_t, float>,
        nb::arg("radius"), nb::arg("height"), nb::arg("segments"),
        "Create a cylinder mesh.");

  m.def("make_cylinder_mesh_int64_double",
        &impl::make_cylinder_mesh_impl<int64_t, double>,
        nb::arg("radius"), nb::arg("height"), nb::arg("segments"),
        "Create a cylinder mesh.");

  // ========== Box mesh (simple) - 4 variants ==========

  m.def("make_box_mesh_int_float",
        &impl::make_box_mesh_impl<int, float>,
        nb::arg("width"), nb::arg("height"), nb::arg("depth"),
        "Create a box mesh.");

  m.def("make_box_mesh_int_double",
        &impl::make_box_mesh_impl<int, double>,
        nb::arg("width"), nb::arg("height"), nb::arg("depth"),
        "Create a box mesh.");

  m.def("make_box_mesh_int64_float",
        &impl::make_box_mesh_impl<int64_t, float>,
        nb::arg("width"), nb::arg("height"), nb::arg("depth"),
        "Create a box mesh.");

  m.def("make_box_mesh_int64_double",
        &impl::make_box_mesh_impl<int64_t, double>,
        nb::arg("width"), nb::arg("height"), nb::arg("depth"),
        "Create a box mesh.");

  // ========== Box mesh (subdivided) - 4 variants ==========

  m.def("make_box_mesh_subdivided_int_float",
        &impl::make_box_mesh_subdivided_impl<int, float>,
        nb::arg("width"), nb::arg("height"), nb::arg("depth"),
        nb::arg("width_ticks"), nb::arg("height_ticks"), nb::arg("depth_ticks"),
        "Create a subdivided box mesh.");

  m.def("make_box_mesh_subdivided_int_double",
        &impl::make_box_mesh_subdivided_impl<int, double>,
        nb::arg("width"), nb::arg("height"), nb::arg("depth"),
        nb::arg("width_ticks"), nb::arg("height_ticks"), nb::arg("depth_ticks"),
        "Create a subdivided box mesh.");

  m.def("make_box_mesh_subdivided_int64_float",
        &impl::make_box_mesh_subdivided_impl<int64_t, float>,
        nb::arg("width"), nb::arg("height"), nb::arg("depth"),
        nb::arg("width_ticks"), nb::arg("height_ticks"), nb::arg("depth_ticks"),
        "Create a subdivided box mesh.");

  m.def("make_box_mesh_subdivided_int64_double",
        &impl::make_box_mesh_subdivided_impl<int64_t, double>,
        nb::arg("width"), nb::arg("height"), nb::arg("depth"),
        nb::arg("width_ticks"), nb::arg("height_ticks"), nb::arg("depth_ticks"),
        "Create a subdivided box mesh.");

  // ========== Plane mesh - 4 variants ==========

  m.def("make_plane_mesh_int_float",
        &impl::make_plane_mesh_impl<int, float>,
        nb::arg("width"), nb::arg("height"),
        nb::arg("width_ticks"), nb::arg("height_ticks"),
        "Create a plane mesh.");

  m.def("make_plane_mesh_int_double",
        &impl::make_plane_mesh_impl<int, double>,
        nb::arg("width"), nb::arg("height"),
        nb::arg("width_ticks"), nb::arg("height_ticks"),
        "Create a plane mesh.");

  m.def("make_plane_mesh_int64_float",
        &impl::make_plane_mesh_impl<int64_t, float>,
        nb::arg("width"), nb::arg("height"),
        nb::arg("width_ticks"), nb::arg("height_ticks"),
        "Create a plane mesh.");

  m.def("make_plane_mesh_int64_double",
        &impl::make_plane_mesh_impl<int64_t, double>,
        nb::arg("width"), nb::arg("height"),
        nb::arg("width_ticks"), nb::arg("height_ticks"),
        "Create a plane mesh.");
}

} // namespace tf::py

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
#include <trueform/geometry/compute_principal_curvatures.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/util/make_numpy_array.hpp>

namespace nb = nanobind;

namespace tf::py {
namespace impl {

// ============================================================================
// Fixed-size mesh (triangles, quads, etc.)
// ============================================================================

template <typename Index, typename RealT, std::size_t Ngon>
auto principal_curvatures(mesh_wrapper<Index, RealT, Ngon, 3> &mesh,
                          std::size_t k, bool directions) {
  auto _polygons = mesh.make_primitive_range();
  auto faces = _polygons.faces();
  auto points = _polygons.points();
  auto tagged_points = points | tf::tag_normals(mesh.point_normals());

  // Make polygons and tag with vertex_link
  // mesh.vertex_link() auto-builds face_membership and vertex_link
  auto polygons =
      tf::make_polygons(faces, tagged_points) | tf::tag(mesh.vertex_link());
  auto n_points = points.size();

  tf::buffer<RealT> k0, k1;
  k0.allocate(n_points);
  k1.allocate(n_points);

  if (directions) {
    tf::unit_vectors_buffer<RealT, 3> d0, d1;
    d0.allocate(n_points);
    d1.allocate(n_points);

    tf::compute_principal_curvatures(polygons, k0, k1, d0, d1, k);

    return nb::make_tuple(make_numpy_array(std::move(k0)),
                          make_numpy_array(std::move(k1)),
                          make_numpy_array(std::move(d0)),
                          make_numpy_array(std::move(d1)));
  } else {
    tf::compute_principal_curvatures(polygons, k0, k1, k);

    return nb::make_tuple(make_numpy_array(std::move(k0)),
                          make_numpy_array(std::move(k1)));
  }
}
} // namespace impl

// ============================================================================
// Registration
// ============================================================================

auto register_principal_curvatures(nb::module_ &m) -> void {
  // ========== Fixed triangles (ngon=3) ==========

  m.def("principal_curvatures_int3float3d",
        &impl::principal_curvatures<int, float, 3>, nb::arg("mesh"),
        nb::arg("k"), nb::arg("directions"),
        "Compute principal curvatures for int32/float32 triangle mesh.");

  m.def("principal_curvatures_int3double3d",
        &impl::principal_curvatures<int, double, 3>, nb::arg("mesh"),
        nb::arg("k"), nb::arg("directions"),
        "Compute principal curvatures for int32/float64 triangle mesh.");

  m.def("principal_curvatures_int643float3d",
        &impl::principal_curvatures<int64_t, float, 3>, nb::arg("mesh"),
        nb::arg("k"), nb::arg("directions"),
        "Compute principal curvatures for int64/float32 triangle mesh.");

  m.def("principal_curvatures_int643double3d",
        &impl::principal_curvatures<int64_t, double, 3>, nb::arg("mesh"),
        nb::arg("k"), nb::arg("directions"),
        "Compute principal curvatures for int64/float64 triangle mesh.");

  // ========== Dynamic mesh (variable n-gons) ==========

  m.def("principal_curvatures_intdynfloat3d",
        &impl::principal_curvatures<int, float, tf::dynamic_size>,
        nb::arg("mesh"), nb::arg("k"), nb::arg("directions"),
        "Compute principal curvatures for int32/float32 dynamic mesh.");

  m.def("principal_curvatures_intdyndouble3d",
        &impl::principal_curvatures<int, double, tf::dynamic_size>,
        nb::arg("mesh"), nb::arg("k"), nb::arg("directions"),
        "Compute principal curvatures for int32/float64 dynamic mesh.");

  m.def("principal_curvatures_int64dynfloat3d",
        &impl::principal_curvatures<int64_t, float, tf::dynamic_size>,
        nb::arg("mesh"), nb::arg("k"), nb::arg("directions"),
        "Compute principal curvatures for int64/float32 dynamic mesh.");

  m.def("principal_curvatures_int64dyndouble3d",
        &impl::principal_curvatures<int64_t, double, tf::dynamic_size>,
        nb::arg("mesh"), nb::arg("k"), nb::arg("directions"),
        "Compute principal curvatures for int64/float64 dynamic mesh.");
}

} // namespace tf::py

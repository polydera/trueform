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
#include <trueform/core/algorithm/parallel_transform.hpp>
#include <trueform/core/angle.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/checked.hpp>
#include <trueform/core/policy/normals.hpp>
#include <trueform/geometry/compute_dihedral_angles.hpp>
#include <trueform/geometry/compute_face_quality.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/topology/policy/manifold_edge_link.hpp>
#include <utility>

namespace nb = nanobind;

namespace tf::py {
namespace impl {

// An angle crosses the boundary as the number it is, so the strong type is
// stripped once, here, and the arrays the caller reads are radians.
template <typename RealT>
auto radians_array(tf::buffer<tf::rad<RealT>> &&angles) {
  tf::buffer<RealT> values;
  values.allocate(angles.size());
  tf::parallel_transform(
      angles, values, [](tf::rad<RealT> angle) { return angle.value; },
      tf::checked);
  return make_numpy_array(std::move(values));
}

template <typename Index, typename RealT, std::size_t Ngon>
auto face_quality(mesh_wrapper<Index, RealT, Ngon, 3> &mesh) {
  auto measured = tf::compute_face_quality(mesh.make_primitive_range());
  return nb::make_tuple(make_numpy_array(std::move(measured.quality)),
                        radians_array(std::move(measured.min_angle)),
                        radians_array(std::move(measured.max_angle)),
                        make_numpy_array(std::move(measured.aspect_ratio)));
}

template <typename Index, typename RealT, std::size_t Ngon>
auto dihedral_angles(mesh_wrapper<Index, RealT, Ngon, 3> &mesh) {
  auto measured = tf::compute_dihedral_angles(
      mesh.make_primitive_range() | tf::tag(mesh.manifold_edge_link()) |
      tf::tag_normals(mesh.normals()));
  return nb::make_tuple(make_numpy_array(std::move(measured.edges)),
                        radians_array(std::move(measured.angles)));
}

} // namespace impl

// ============================================================================
// Registration
// ============================================================================

auto register_quality(nb::module_ &m) -> void {

  // ========== face_quality ==========

  m.def("face_quality_int3float3d", &impl::face_quality<int, float, 3>,
        nb::arg("mesh"));
  m.def("face_quality_int3double3d", &impl::face_quality<int, double, 3>,
        nb::arg("mesh"));
  m.def("face_quality_int643float3d", &impl::face_quality<int64_t, float, 3>,
        nb::arg("mesh"));
  m.def("face_quality_int643double3d", &impl::face_quality<int64_t, double, 3>,
        nb::arg("mesh"));
  m.def("face_quality_intdynfloat3d",
        &impl::face_quality<int, float, tf::dynamic_size>, nb::arg("mesh"));
  m.def("face_quality_intdyndouble3d",
        &impl::face_quality<int, double, tf::dynamic_size>, nb::arg("mesh"));
  m.def("face_quality_int64dynfloat3d",
        &impl::face_quality<int64_t, float, tf::dynamic_size>, nb::arg("mesh"));
  m.def("face_quality_int64dyndouble3d",
        &impl::face_quality<int64_t, double, tf::dynamic_size>,
        nb::arg("mesh"));

  // ========== dihedral_angles ==========

  m.def("dihedral_angles_int3float3d", &impl::dihedral_angles<int, float, 3>,
        nb::arg("mesh"));
  m.def("dihedral_angles_int3double3d", &impl::dihedral_angles<int, double, 3>,
        nb::arg("mesh"));
  m.def("dihedral_angles_int643float3d",
        &impl::dihedral_angles<int64_t, float, 3>, nb::arg("mesh"));
  m.def("dihedral_angles_int643double3d",
        &impl::dihedral_angles<int64_t, double, 3>, nb::arg("mesh"));
  m.def("dihedral_angles_intdynfloat3d",
        &impl::dihedral_angles<int, float, tf::dynamic_size>, nb::arg("mesh"));
  m.def("dihedral_angles_intdyndouble3d",
        &impl::dihedral_angles<int, double, tf::dynamic_size>, nb::arg("mesh"));
  m.def("dihedral_angles_int64dynfloat3d",
        &impl::dihedral_angles<int64_t, float, tf::dynamic_size>,
        nb::arg("mesh"));
  m.def("dihedral_angles_int64dyndouble3d",
        &impl::dihedral_angles<int64_t, double, tf::dynamic_size>,
        nb::arg("mesh"));
}

} // namespace tf::py

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
#pragma once

#include "../spatial/mesh.hpp"
#include "./volume_ops_impl.hpp"
#include "./volume_real_request.hpp"
#include "./volume_wrapper.hpp"
#include <array>
#include <trueform/core/form.hpp>
#include <trueform/core/frame.hpp>
#include <trueform/core/policy/frame.hpp>
#include <trueform/spatial/policy/tree.hpp>
#include <trueform/volume/make_mesh_sdf.hpp>
#include <trueform/volume/mesh_sdf_config.hpp>

namespace tf::py {

// The grid samples the mesh in the frame it states: world space when the
// wrapper carries a transformation, the local frame otherwise. The caller's
// dims/spacing/origin describe the grid in that same space.
template <typename Index, typename RealT, std::size_t Ngon>
auto make_mesh_sdf_impl(mesh_wrapper<Index, RealT, Ngon, 3> &mesh,
                        std::array<int, 3> dims, std::array<double, 3> spacing,
                        std::array<double, 3> origin, int dtype, int mode,
                        int band) {
  mesh.build_tree();
  auto form = mesh.make_primitive_range() | tf::tag(mesh.tree());
  const tf::mesh_sdf_config config{
      mode == 1 ? tf::mesh_sdf_mode::banded : tf::mesh_sdf_mode::exact, band};
  return with_requested_real(dtype, [&](auto real) {
    using Real = decltype(real);
    const auto sp = to_volume_point<Real>(spacing);
    const auto og = to_volume_point<Real>(origin);
    if (mesh.has_transformation())
      return to_python_volume(tf::make_mesh_sdf<Real>(
          form | tf::tag(tf::make_frame(mesh.transformation_view())), dims, sp,
          og, config));
    return to_python_volume(
        tf::make_mesh_sdf<Real>(form, dims, sp, og, config));
  });
}

template <typename Index, typename RealT, std::size_t Ngon>
auto def_make_mesh_sdf(nanobind::module_ &m, const char *name) -> void {
  m.def(name, &make_mesh_sdf_impl<Index, RealT, Ngon>, nanobind::arg("mesh"),
        nanobind::arg("dims"), nanobind::arg("spacing"), nanobind::arg("origin"),
        nanobind::arg("dtype"), nanobind::arg("mode"), nanobind::arg("band"),
        "Sample the signed distance field of a closed mesh onto a regular "
        "grid (exact crossing parity, negative inside by winding), in the "
        "frame the mesh states; mode 1 measures a band and sweeps the far "
        "field.");
}

} // namespace tf::py

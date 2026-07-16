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
#include "../intersect/build_intersect_structures.hpp"
#include "../spatial/mesh.hpp"
#include "../util/make_numpy_array.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <trueform/csg/make_outer_shell.hpp>
#include <trueform/topology/domain_config.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include <trueform/topology/policy/manifold_edge_link.hpp>

namespace tf::py {

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto outer_shell(mesh_wrapper<Index, RealT, Ngon, Dims> &form_wrapper,
                 int domain_flags) {
  build_intersect_structures(form_wrapper);
  auto form = form_wrapper.make_primitive_range() |
              tf::tag(form_wrapper.manifold_edge_link()) |
              tf::tag(form_wrapper.face_membership()) |
              tf::tag(form_wrapper.tree());
  auto shell = tf::make_outer_shell(
      form,
      tf::intersect_config{tf::intersect_mode::primitives |
                           tf::intersect_mode::resolve_contours},
      static_cast<tf::domain_config>(domain_flags));
  return make_numpy_array(std::move(shell));
}

} // namespace tf::py

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
#include "../util/make_numpy_array.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <trueform/cut/embedded_self_intersection_curves.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include <trueform/topology/policy/manifold_edge_link.hpp>

namespace tf::py {
template <typename Index0, typename RealT, std::size_t Ngon0, std::size_t Dims>
auto embedded_self_intersection_curves(
    mesh_wrapper<Index0, RealT, Ngon0, Dims> &form_wrapper) {
  auto form0 = form_wrapper.make_primitive_range() |
               tf::tag(form_wrapper.manifold_edge_link()) |
               tf::tag(form_wrapper.face_membership()) |
               tf::tag(form_wrapper.tree());
  auto [faces, points] =
      make_numpy_array(tf::embedded_self_intersection_curves(form0));
  return nanobind::make_tuple(faces, points);
}

template <typename Index0, typename RealT, std::size_t Ngon0, std::size_t Dims>
auto embedded_self_intersection_curves(
    mesh_wrapper<Index0, RealT, Ngon0, Dims> &form_wrapper,
    tf::return_curves_t) {
  auto form0 = form_wrapper.make_primitive_range() |
               tf::tag(form_wrapper.manifold_edge_link()) |
               tf::tag(form_wrapper.face_membership()) |
               tf::tag(form_wrapper.tree());
  auto [polygons, curves] =
      tf::embedded_self_intersection_curves(form0, tf::return_curves);
  auto [faces, points] = make_numpy_array(std::move(polygons));
  auto [paths, c_points] = make_numpy_array(std::move(curves));
  return nanobind::make_tuple(
      nanobind::make_tuple(faces, points),
      nanobind::make_tuple(nanobind::make_tuple(paths.first, paths.second),
                           std::move(c_points)));
}
} // namespace tf::py

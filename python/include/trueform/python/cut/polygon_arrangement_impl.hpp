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
#include <trueform/cut/make_polygon_arrangements.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include <trueform/topology/policy/manifold_edge_link.hpp>

namespace tf::py {

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto polygon_arrangements(
    mesh_wrapper<Index, RealT, Ngon, Dims> &form_wrapper, int mode,
    double tolerance) {
  auto form = form_wrapper.make_primitive_range() |
              tf::tag(form_wrapper.manifold_edge_link()) |
              tf::tag(form_wrapper.face_membership()) |
              tf::tag(form_wrapper.tree());
  auto [mesh, face_labels] = tf::make_polygon_arrangements(
      form,
      tf::intersect_config{static_cast<tf::intersect_mode>(mode), tolerance});
  auto mesh_pair = make_numpy_array(std::move(mesh));
  auto labels_array = make_numpy_array(std::move(face_labels));
  return nanobind::make_tuple(std::move(mesh_pair), std::move(labels_array));
}

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto polygon_arrangements(
    mesh_wrapper<Index, RealT, Ngon, Dims> &form_wrapper, int mode,
    double tolerance, tf::return_curves_t) {
  auto form = form_wrapper.make_primitive_range() |
              tf::tag(form_wrapper.manifold_edge_link()) |
              tf::tag(form_wrapper.face_membership()) |
              tf::tag(form_wrapper.tree());
  auto [mesh, face_labels, curves] = tf::make_polygon_arrangements(
      form,
      tf::intersect_config{static_cast<tf::intersect_mode>(mode), tolerance},
      tf::return_curves);
  auto mesh_pair = make_numpy_array(std::move(mesh));
  auto labels_array = make_numpy_array(std::move(face_labels));
  auto [paths, c_points] = make_numpy_array(std::move(curves));
  return nanobind::make_tuple(
      std::move(mesh_pair), std::move(labels_array),
      nanobind::make_tuple(nanobind::make_tuple(paths.first, paths.second),
                           std::move(c_points)));
}

} // namespace tf::py

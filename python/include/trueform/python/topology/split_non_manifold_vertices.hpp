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
#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <trueform/reindex/return_index_map.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include <trueform/topology/split_non_manifold_vertices.hpp>
#include <utility>

namespace tf::py {
template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto split_non_manifold_vertices(
    mesh_wrapper<Index, RealT, Ngon, Dims> &form_wrapper) {
  auto split = tf::split_non_manifold_vertices(
      form_wrapper.make_primitive_range() |
          tf::tag(form_wrapper.face_membership()),
      tf::return_index_map);
  auto mesh = make_numpy_array(std::move(split.first));
  return nanobind::make_tuple(mesh.first, mesh.second,
                              make_numpy_array(std::move(split.second)));
}
} // namespace tf::py

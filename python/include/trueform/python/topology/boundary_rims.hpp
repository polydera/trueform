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
#include "../core/offset_blocked_array.hpp"
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/algorithm/parallel_transform.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/checked.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/topology/boundary_rims.hpp>
#include <utility>

namespace tf::py {
template <typename Index>
auto make_boundary_rims_tuple(tf::boundary_rims<Index> &&rims) {
  tf::buffer<std::int8_t> closed;
  closed.allocate(rims.closed.size());
  tf::parallel_transform(
      rims.closed, closed,
      [](bool closes) { return static_cast<std::int8_t>(closes); },
      tf::checked);
  auto [vertex_offsets, vertex_data] =
      make_numpy_array(std::move(rims.vertices));
  auto [face_offsets, face_data] = make_numpy_array(std::move(rims.faces));
  return nanobind::make_tuple(
      offset_blocked_array_wrapper<Index, Index>{vertex_offsets, vertex_data},
      offset_blocked_array_wrapper<Index, Index>{face_offsets, face_data},
      make_numpy_array(std::move(closed)));
}

template <typename Index, std::size_t Ngon>
auto boundary_rims(
    nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>> cells,
    const offset_blocked_array_wrapper<Index, Index> &fm) {
  auto faces = tf::make_faces(
      tf::make_blocked_range<Ngon>(tf::make_range(cells.data(), cells.size())));
  auto fml = tf::make_face_membership_like(fm.make_range());
  return make_boundary_rims_tuple(tf::make_boundary_rims(faces, fml));
}

template <typename Index>
auto boundary_rims_dynamic(
    const offset_blocked_array_wrapper<Index, Index> &cells,
    const offset_blocked_array_wrapper<Index, Index> &fm) {
  auto faces = tf::make_faces(cells.make_range());
  auto fml = tf::make_face_membership_like(fm.make_range());
  return make_boundary_rims_tuple(tf::make_boundary_rims(faces, fml));
}
} // namespace tf::py

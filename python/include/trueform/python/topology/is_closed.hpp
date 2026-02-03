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
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/topology/is_closed.hpp>

namespace tf::py {

template <typename Index, std::size_t Ngon>
auto is_closed(
    nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>> cells,
    const offset_blocked_array_wrapper<Index, Index> &fm) -> bool {
  auto faces = tf::make_faces(
      tf::make_blocked_range<Ngon>(tf::make_range(cells.data(), cells.size())));
  auto fml = tf::make_face_membership_like(fm.make_range());
  return tf::is_closed(faces, fml);
}

template <typename Index>
auto is_closed_dynamic(
    const offset_blocked_array_wrapper<Index, Index> &cells,
    const offset_blocked_array_wrapper<Index, Index> &fm) -> bool {
  auto faces = tf::make_faces(cells.make_range());
  auto fml = tf::make_face_membership_like(fm.make_range());
  return tf::is_closed(faces, fml);
}

} // namespace tf::py

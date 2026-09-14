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

#include "trueform/core/faces.hpp"
#include "trueform/core/range.hpp"
#include "trueform/core/static_size.hpp"

#include <cstddef>
#include <utility>

namespace tf::cpp::detail {

template <typename Index, std::size_t Ngon> struct mesh_faces_view;

template <typename Index> struct mesh_faces_view<Index, 3> {
  using type = decltype(tf::make_faces<3>(
      std::declval<tf::range<const Index *, tf::dynamic_size>>()));
};

template <typename Index> struct mesh_faces_view<Index, tf::dynamic_size> {
  using type = decltype(tf::make_faces(
      std::declval<tf::range<const Index *, tf::dynamic_size>>(),
      std::declval<tf::range<const Index *, tf::dynamic_size>>()));
};

} // namespace tf::cpp::detail

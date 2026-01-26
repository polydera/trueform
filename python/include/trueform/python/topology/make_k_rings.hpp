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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/python/core/offset_blocked_array.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/topology/make_k_rings.hpp>
#include <trueform/topology/vertex_link_like.hpp>

namespace tf::py {

template <typename Index>
auto make_k_rings(const offset_blocked_array_wrapper<Index, Index> &connectivity,
                  std::size_t k, bool inclusive = false) {

  auto vl = tf::make_vertex_link_like(connectivity.make_range());
  auto result = tf::make_k_rings(vl, k, inclusive);
  auto [offsets, data] = make_numpy_array(std::move(result));
  return offset_blocked_array_wrapper<Index, Index>{offsets, data};
}

} // namespace tf::py

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
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/topology/connect_edges_to_paths.hpp>

namespace tf::py {
template <typename Index>
auto connect_edges_to_paths(
    nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, 2>> array) {
  auto edges = tf::make_edges(
      tf::make_blocked_range<2>(tf::make_range(array.data(), array.size())));
  auto [offsets, data] = make_numpy_array(tf::connect_edges_to_paths(edges));
  return nanobind::make_tuple(std::move(offsets), std::move(data));
}
} // namespace tf::py

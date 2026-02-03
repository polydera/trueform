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
#include <trueform/core/points.hpp>
#include <trueform/geometry/laplacian_smoothed.hpp>
#include <trueform/python/core/offset_blocked_array.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/topology/vertex_link_like.hpp>

namespace tf::py {

template <typename Index, typename RealT, std::size_t Dims>
auto laplacian_smoothed(
    nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>> points,
    const offset_blocked_array_wrapper<Index, Index> &vertex_link,
    std::size_t iterations, RealT lambda) {

  auto vl = tf::make_vertex_link_like(vertex_link.make_range());

  RealT *pts_data = static_cast<RealT *>(points.data());
  auto flat_range = tf::make_range(pts_data, points.shape(0) * Dims);
  auto pts = tf::make_points<Dims>(flat_range) | tf::tag(vl);

  auto result = tf::laplacian_smoothed(pts, iterations, lambda);

  return make_numpy_array(std::move(result));
}

} // namespace tf::py

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
#include <nanobind/stl/tuple.h>

#include <trueform/core/points.hpp>
#include <trueform/core/polygons.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/python/core/offset_blocked_array.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/topology/domain_config.hpp>
#include <trueform/topology/make_domain_labels.hpp>

namespace tf::py::impl {

template <typename Index, std::size_t V, typename RealT, std::size_t Dims>
auto domain_labels_impl(
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1, V>>
        indices,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    int config) {
  auto i_range =
      tf::make_blocked_range<V>(tf::make_range(indices.data(), indices.size()));
  auto p_range =
      tf::make_points<Dims>(tf::make_range(points.data(), points.size()));
  auto polygons = tf::make_polygons(i_range, p_range);

  auto dl = tf::make_domain_labels(polygons,
                                   static_cast<tf::domain_config>(config));

  return nanobind::make_tuple(
      make_numpy_array(std::move(dl.labels.data_buffer())), dl.n_domains,
      dl.outer_shell_label);
}

template <typename Index, typename RealT, std::size_t Dims>
auto domain_labels_impl_dynamic(
    const offset_blocked_array_wrapper<Index, Index> &indices,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    int config) {
  auto p_range =
      tf::make_points<Dims>(tf::make_range(points.data(), points.size()));
  auto i_range = indices.make_range();
  auto polygons = tf::make_polygons(i_range, p_range);

  auto dl = tf::make_domain_labels(polygons,
                                   static_cast<tf::domain_config>(config));

  return nanobind::make_tuple(
      make_numpy_array(std::move(dl.labels.data_buffer())), dl.n_domains,
      dl.outer_shell_label);
}

} // namespace tf::py::impl

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
#include <nanobind/stl/tuple.h>

#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/topology/domain_config.hpp>
#include <trueform/topology/make_domain_labels.hpp>

namespace tf::py::impl {

template <typename Index, std::size_t Ngon, typename RealT, std::size_t Dims>
auto domain_labels_impl(mesh_wrapper<Index, RealT, Ngon, Dims> &wrapper,
                        int config) {
  auto polygons = wrapper.make_primitive_range() |
                  tf::tag(wrapper.face_membership()) |
                  tf::tag(wrapper.manifold_edge_link());

  auto dl = tf::make_domain_labels(polygons,
                                   static_cast<tf::domain_config>(config));

  return nanobind::make_tuple(
      make_numpy_array(std::move(dl.labels.data_buffer())), dl.n_domains,
      dl.outer_shell_label);
}

} // namespace tf::py::impl

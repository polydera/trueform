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
#include "trueform/core/views/blocked_range.hpp"
#include "trueform/python/util/make_numpy_array.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/buffer.hpp>
#include <trueform/topology/label_connected_components.hpp>

namespace tf::py {
namespace impl {
template <typename Index, typename Range>
auto label_connected_components_impl(
    const Range &conn, std::optional<Index> expected_number_of_components) {
  tf::buffer<int> labels;
  labels.allocate(conn.size());
  Index enc = 2;
  if (expected_number_of_components)
    enc = *expected_number_of_components;
  int n_components = tf::label_connected_components<Index>(
      labels,
      [&](Index i, auto &&f) {
        for (const auto &next : conn[i]) {
          if (next >= 0)
            f(next);
        }
      },
      enc);
  return nanobind::make_tuple(n_components,
                              make_numpy_array(std::move(labels)));
}
} // namespace impl
template <typename Index>
auto label_connected_components_impl(
    const tf::py::offset_blocked_array_wrapper<Index, Index> &conn,
    std::optional<Index> expected_number_of_components) {
  return impl::label_connected_components_impl(conn.make_range(),
                                               expected_number_of_components);
}

template <typename Index>
auto label_connected_components_impl(
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1, -1>>
        conn,
    std::optional<Index> expected_number_of_components) {
  auto size = conn.shape(0);
  auto block_size = conn.shape(1);
  return impl::label_connected_components_impl(
      tf::make_blocked_range(tf::make_range(conn.data(), size * block_size),
                             block_size),
      expected_number_of_components);
}
} // namespace tf::py

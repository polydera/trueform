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
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/python/core/offset_blocked_array.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/topology/edge_membership.hpp>
#include <trueform/topology/face_membership.hpp>

namespace tf::py {
template <typename Index, std::size_t Ngon>
auto compute_cell_membership(
    nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>> cells,
    Index n_ids) {
  auto membership = [&] {
    if constexpr (Ngon == 2) {
      tf::edge_membership<Index> fm;
      fm.build(tf::make_edges(tf::make_blocked_range<Ngon>(
                   tf::make_range(cells.data(), cells.size()))),
               n_ids);
      return fm;
    } else {
      tf::face_membership<Index> fm;
      fm.build(tf::make_faces(tf::make_blocked_range<Ngon>(
                   tf::make_range(cells.data(), cells.size()))),
               n_ids, Index(cells.size()));
      return fm;
    }
  }();
  auto [offsets, data] = make_numpy_array(std::move(membership));
  return offset_blocked_array_wrapper<Index, Index>{offsets, data};
}

template <typename Index>
auto compute_cell_membership_dynamic(
    const offset_blocked_array_wrapper<Index, Index> &cells, Index n_ids) {
  tf::face_membership<Index> fm;
  fm.build(tf::make_faces(cells.make_range()), n_ids,
           Index(cells.data_array().size()));
  auto [offsets, data] = make_numpy_array(std::move(fm));
  return offset_blocked_array_wrapper<Index, Index>{offsets, data};
}
} // namespace tf::py

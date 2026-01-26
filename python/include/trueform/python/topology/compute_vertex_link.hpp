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
#include <trueform/topology/vertex_link.hpp>

namespace tf::py {
template <typename Index, std::size_t Ngon>
auto compute_vertex_link(
    nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>>
        cell_array,
    const offset_blocked_array_wrapper<Index, Index> &face_membership) {

  Index *data_fcs = static_cast<Index *>(cell_array.data());
  std::size_t count_fcs = cell_array.shape(0) * Ngon;
  auto faces = tf::make_faces(
      tf::make_blocked_range<Ngon>(tf::make_range(data_fcs, count_fcs)));

  auto fm = tf::make_face_membership_like(face_membership.make_range());

  tf::vertex_link<Index> fl;
  fl.build(faces, fm);
  auto [offsets, data] = make_numpy_array(std::move(fl));
  return offset_blocked_array_wrapper<Index, Index>{offsets, data};
}

template <typename Index>
auto compute_vertex_link(
    nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, 2>>
        cell_array,
    Index n_ids) {

  Index *data_fcs = static_cast<Index *>(cell_array.data());
  std::size_t count_fcs = cell_array.shape(0) * 2;
  auto edges = tf::make_edges(
      tf::make_blocked_range<2>(tf::make_range(data_fcs, count_fcs)));

  tf::vertex_link<Index> fl;
  fl.build(edges, n_ids);
  auto [offsets, data] = make_numpy_array(std::move(fl));
  return offset_blocked_array_wrapper<Index, Index>{offsets, data};
}

template <typename Index>
auto compute_vertex_link_dynamic(
    const offset_blocked_array_wrapper<Index, Index> &faces_array,
    const offset_blocked_array_wrapper<Index, Index> &face_membership) {

  auto faces = tf::make_faces(faces_array.make_range());
  auto fm = tf::make_face_membership_like(face_membership.make_range());

  tf::vertex_link<Index> fl;
  fl.build(faces, fm);
  auto [offsets, data] = make_numpy_array(std::move(fl));
  return offset_blocked_array_wrapper<Index, Index>{offsets, data};
}
} // namespace tf::py

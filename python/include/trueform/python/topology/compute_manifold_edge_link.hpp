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
#include <trueform/core/algorithm/parallel_copy.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/python/core/offset_blocked_array.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/topology/manifold_edge_link.hpp>

namespace tf::py {
template <typename Index, std::size_t Ngon>
auto compute_manifold_edge_link(
    nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>>
        faces_array,
    const offset_blocked_array_wrapper<Index, Index> &face_membership) {

  Index *data_fcs = static_cast<Index *>(faces_array.data());
  std::size_t count_fcs = faces_array.shape(0) * Ngon;
  auto faces =
      tf::make_blocked_range<Ngon>(tf::make_range(data_fcs, count_fcs));

  auto fm = tf::make_face_membership_like(face_membership.make_range());

  tf::blocked_buffer<Index, Ngon> buff;
  buff.allocate(faces.size());
  tf::topology::compute_manifold_edge_link<Index>(faces, fm, buff);
  return make_numpy_array(std::move(buff));
}

template <typename Index>
auto compute_manifold_edge_link_dynamic(
    const offset_blocked_array_wrapper<Index, Index> &faces_array,
    const offset_blocked_array_wrapper<Index, Index> &face_membership) {

  auto faces = faces_array.make_range();
  auto fm = tf::make_face_membership_like(face_membership.make_range());

  tf::offset_block_buffer<Index, Index> buff;
  buff.offsets_buffer().allocate(faces_array.offsets_array().size());
  buff.data_buffer().allocate(faces_array.data_array().size());

  // Copy offsets from faces
  const Index *faces_offsets =
      static_cast<const Index *>(faces_array.offsets_array().data());
  tf::parallel_copy(
      tf::make_range(faces_offsets, faces_array.offsets_array().size()),
      buff.offsets_buffer());

  // Compute manifold edge link into the buffer
  tf::topology::compute_manifold_edge_link<Index>(faces, fm, buff);
  auto [offsets, data] = make_numpy_array(std::move(buff));
  return offset_blocked_array_wrapper<Index, Index>{offsets, data};
}
} // namespace tf::py

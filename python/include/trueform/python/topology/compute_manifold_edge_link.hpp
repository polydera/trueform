/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
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
} // namespace tf::py

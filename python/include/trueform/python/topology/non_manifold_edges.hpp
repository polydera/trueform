/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include "../core/offset_blocked_array.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/topology/non_manifold_edges.hpp>

namespace tf::py {
template <typename Index, std::size_t Ngon>
auto non_manifold_edges(
    nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>> cells,
    const offset_blocked_array_wrapper<Index, Index> &fm) {
  auto faces = tf::make_faces(
      tf::make_blocked_range<Ngon>(tf::make_range(cells.data(), cells.size())));
  auto fml = tf::make_face_membership_like(fm.make_range());
  return make_numpy_array(tf::make_non_manifold_edges(faces, fml));
}
} // namespace tf::py

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
#include <trueform/geometry/ensure_positive_orientation.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/topology/policy/manifold_edge_link.hpp>

namespace tf::py {

template <typename Index, typename RealT, std::size_t Ngon>
auto ensure_positive_orientation(mesh_wrapper<Index, RealT, Ngon, 3> &mesh,
                                 bool is_consistent = false) -> bool {
  // Reuse manifold_edge_link from mesh wrapper (builds if not cached)
  auto polygons =
      mesh.make_primitive_range() | tf::tag(mesh.manifold_edge_link());
  return tf::ensure_positive_orientation(polygons, is_consistent);
}

} // namespace tf::py

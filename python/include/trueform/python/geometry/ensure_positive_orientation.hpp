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
#include <trueform/core/signed_volume.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/topology/orient_faces_consistently.hpp>
#include <trueform/topology/reverse_winding.hpp>

namespace tf::py {

template <typename Index, typename RealT, std::size_t Ngon>
auto ensure_positive_orientation(mesh_wrapper<Index, RealT, Ngon, 3> &mesh,
                                  bool is_consistent = false) {
  if (!is_consistent) {
    // Reuse manifold_edge_link from mesh wrapper (builds if not cached)
    tf::orient_faces_consistently(mesh.make_primitive_range() |
                                  tf::tag(mesh.manifold_edge_link()));
  }
  // Check signed volume and reverse if negative
  auto polygons = mesh.make_primitive_range();
  if (tf::signed_volume(polygons) < 0) {
    tf::reverse_winding(polygons.faces());
  }
}

} // namespace tf::py

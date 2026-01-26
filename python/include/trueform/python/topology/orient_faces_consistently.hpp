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
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/topology/orient_faces_consistently.hpp>
#include <trueform/topology/policy/manifold_edge_link.hpp>

namespace tf::py {
template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto orient_faces_consistently(mesh_wrapper<Index, RealT, Ngon, Dims> &mesh) {
  tf::orient_faces_consistently(mesh.make_primitive_range() |
                                tf::tag(mesh.manifold_edge_link()));
}
} // namespace tf::py

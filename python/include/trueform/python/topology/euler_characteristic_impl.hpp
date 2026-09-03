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

#include <trueform/python/spatial/mesh.hpp>
#include <trueform/topology/euler_characteristic.hpp>
#include <trueform/topology/policy/manifold_edge_link.hpp>

namespace tf::py::impl {

template <typename Index, std::size_t Ngon, typename RealT, std::size_t Dims>
auto euler_characteristic_impl(mesh_wrapper<Index, RealT, Ngon, Dims> &wrapper)
    -> int {
  auto polygons = wrapper.make_primitive_range() |
                  tf::tag(wrapper.manifold_edge_link());
  return tf::euler_characteristic(polygons);
}

} // namespace tf::py::impl

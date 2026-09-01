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
#include "../arrangement/arrangement_builders.hpp"
#include "../spatial/mesh.hpp"
#include "../util/make_numpy_array.hpp"
#include "./build_intersect_structures.hpp"
#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <trueform/core/transformation.hpp>
#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/arrangement/make_intersection_curves.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <utility>

namespace tf::py {
template <typename Index0, typename RealT, std::size_t Ngon0, std::size_t Dims>
auto self_intersection_curves(
    mesh_wrapper<Index0, RealT, Ngon0, Dims> &form_wrapper,
    tf::intersect_mode mode = tf::intersect_mode::primitives,
    double tolerance = 0.0) {
  build_intersect_structures(form_wrapper);
  // A self arrangement runs in the mesh's own coordinates: this entry has
  // never carried the wrapper's transformation.
  auto graph = build_self_arrangement(
      tagged_form(form_wrapper,
                  tf::make_identity_transformation<RealT, Dims>()),
      tf::arrangement_config{tf::intersect_config{mode, tolerance}});
  auto curves = tf::make_intersection_curves<RealT>(graph);
  auto [paths, c_points] = make_numpy_array(std::move(curves));
  return nanobind::make_tuple(nanobind::make_tuple(paths.first, paths.second),
                              std::move(c_points));
}
} // namespace tf::py

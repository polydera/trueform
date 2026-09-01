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
#include "../intersect/build_intersect_structures.hpp"
#include "../spatial/mesh.hpp"
#include "../util/make_numpy_array.hpp"
#include "./csg_builders.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <trueform/core/range.hpp>
#include <trueform/core/transformation.hpp>
#include <trueform/csg/make_outer_shell.hpp>
#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/intersect/intersect_mode.hpp>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::py {

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto outer_shell(mesh_wrapper<Index, RealT, Ngon, Dims> &form_wrapper) {
  build_intersect_structures(form_wrapper);
  // The repair runs in the mesh's own coordinates: this entry has never
  // carried the wrapper's transformation. The forms outlive the graph
  // that views them.
  std::array<form_t<Index, RealT, Ngon, Dims>, 1> forms = {
      tagged_form(form_wrapper,
                  tf::make_identity_transformation<RealT, Dims>())};
  const int *no_sheets = nullptr;
  auto graph = build_range_csg_graph(
      tf::make_range(forms.data(), forms.size()),
      tf::make_range(no_sheets, no_sheets),
      tf::arrangement_config{
          tf::intersect_config{tf::intersect_mode::primitives |
                               tf::intersect_mode::resolve_contours}});
  auto shell = tf::make_outer_shell(graph);
  return make_numpy_array(std::move(shell));
}

} // namespace tf::py

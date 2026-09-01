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
#include "../intersect/build_intersect_structures.hpp"
#include "../spatial/mesh.hpp"
#include "../util/make_numpy_array.hpp"
#include "./arrangement_builders.hpp"
#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <trueform/core/none.hpp>
#include <trueform/core/range.hpp>
#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/arrangement/make_mesh_arrangements.hpp>
#include <trueform/arrangement/return_curves.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/topology/triangulation_type.hpp>
#include <utility>
#include <vector>

namespace tf::py {

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto mesh_arrangements(
    std::vector<mesh_wrapper<Index, RealT, Ngon, Dims>> &wrappers, int mode,
    double tolerance, int triangulation) {
  build_intersect_structures_all(wrappers);
  auto forms = tagged_forms(wrappers);
  auto graph = build_range_arrangement(
      tf::make_range(forms.data(), forms.size()),
      tf::arrangement_config{
          tf::intersect_config{static_cast<tf::intersect_mode>(mode),
                               tolerance},
          static_cast<tf::triangulation_type>(triangulation)});
  auto [mesh, tag_labels, face_labels] =
      tf::arrangement::arrangement_worker<tf::none_t, tf::none_t, tf::none_t>(
          graph);
  return nanobind::make_tuple(make_numpy_array(std::move(mesh)),
                              make_numpy_array(std::move(tag_labels)),
                              make_numpy_array(std::move(face_labels)));
}

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto mesh_arrangements(
    std::vector<mesh_wrapper<Index, RealT, Ngon, Dims>> &wrappers, int mode,
    double tolerance, int triangulation, tf::return_curves_t) {
  build_intersect_structures_all(wrappers);
  auto forms = tagged_forms(wrappers);
  auto graph = build_range_arrangement(
      tf::make_range(forms.data(), forms.size()),
      tf::arrangement_config{
          tf::intersect_config{static_cast<tf::intersect_mode>(mode),
                               tolerance},
          static_cast<tf::triangulation_type>(triangulation)});
  auto [mesh, tag_labels, face_labels, curves] =
      tf::arrangement::arrangement_worker<tf::none_t, tf::return_curves_t,
                                  tf::none_t>(graph);
  auto mesh_pair = make_numpy_array(std::move(mesh));
  auto tag_array = make_numpy_array(std::move(tag_labels));
  auto face_array = make_numpy_array(std::move(face_labels));
  auto [paths, c_points] = make_numpy_array(std::move(curves));
  auto curve_tuple = nanobind::make_tuple(
      nanobind::make_tuple(paths.first, paths.second), std::move(c_points));
  return nanobind::make_tuple(std::move(mesh_pair), std::move(tag_array),
                              std::move(face_array), std::move(curve_tuple));
}

} // namespace tf::py

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
#include <trueform/core/none.hpp>
#include <trueform/core/transformation.hpp>
#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/arrangement/make_mesh_arrangements.hpp>
#include <trueform/arrangement/return_curves.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/topology/triangulation_type.hpp>
#include <utility>

namespace tf::py {

namespace detail {
// A single-mesh arrangement runs in the mesh's own coordinates: this
// entry has never carried the wrapper's transformation.
template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto polygon_arrangement_graph(
    mesh_wrapper<Index, RealT, Ngon, Dims> &form_wrapper, int mode,
    double tolerance, int triangulation)
    -> self_arrangement_t<form_t<Index, RealT, Ngon, Dims>> {
  build_intersect_structures(form_wrapper);
  return build_self_arrangement(
      tagged_form(form_wrapper,
                  tf::make_identity_transformation<RealT, Dims>()),
      tf::arrangement_config{
          tf::intersect_config{static_cast<tf::intersect_mode>(mode),
                               tolerance},
          static_cast<tf::triangulation_type>(triangulation)});
}
} // namespace detail

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto polygon_arrangements(
    mesh_wrapper<Index, RealT, Ngon, Dims> &form_wrapper, int mode,
    double tolerance, int triangulation) {
  auto graph = detail::polygon_arrangement_graph(form_wrapper, mode, tolerance,
                                                 triangulation);
  auto [mesh, face_labels] =
      tf::arrangement::arrangement_worker<tf::none_t, tf::none_t, tf::none_t>(
          graph);
  auto mesh_pair = make_numpy_array(std::move(mesh));
  auto labels_array = make_numpy_array(std::move(face_labels));
  return nanobind::make_tuple(std::move(mesh_pair), std::move(labels_array));
}

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto polygon_arrangements(
    mesh_wrapper<Index, RealT, Ngon, Dims> &form_wrapper, int mode,
    double tolerance, int triangulation, tf::return_curves_t) {
  auto graph = detail::polygon_arrangement_graph(form_wrapper, mode, tolerance,
                                                 triangulation);
  auto [mesh, face_labels, curves] =
      tf::arrangement::arrangement_worker<tf::none_t, tf::return_curves_t,
                                  tf::none_t>(graph);
  auto mesh_pair = make_numpy_array(std::move(mesh));
  auto labels_array = make_numpy_array(std::move(face_labels));
  auto [paths, c_points] = make_numpy_array(std::move(curves));
  return nanobind::make_tuple(
      std::move(mesh_pair), std::move(labels_array),
      nanobind::make_tuple(nanobind::make_tuple(paths.first, paths.second),
                           std::move(c_points)));
}

} // namespace tf::py

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
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <trueform/core/range.hpp>
#include <trueform/csg/boolean_op.hpp>
#include <trueform/csg/expression/make_boolean_expr.hpp>
#include <trueform/csg/graph/make_boolean_labels.hpp>
#include <trueform/csg/make_csg_mesh.hpp>
#include <trueform/csg/make_intersection_curves.hpp>
#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/arrangement/return_curves.hpp>
#include <trueform/reindex/return_source_ids.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tf::py {

namespace detail {
template <typename Index0, typename RealT, std::size_t Ngon0,
          std::size_t Dims, typename Index1, std::size_t Ngon1>
auto boolean_graph(mesh_wrapper<Index0, RealT, Ngon0, Dims> &form_wrapper0,
                   mesh_wrapper<Index1, RealT, Ngon1, Dims> &form_wrapper1,
                   const std::vector<int> &sheets)
    -> pair_csg_graph_t<form_t<Index0, RealT, Ngon0, Dims>,
                        form_t<Index1, RealT, Ngon1, Dims>> {
  build_intersect_structures(form_wrapper0, form_wrapper1);
  return build_pair_csg_graph(
      tagged_form(form_wrapper0, form_transformation(form_wrapper0)),
      tagged_form(form_wrapper1, form_transformation(form_wrapper1)),
      tf::make_range(sheets.data(), sheets.size()), tf::arrangement_config{});
}
} // namespace detail

template <typename Index0, typename RealT, std::size_t Ngon0, std::size_t Dims,
          typename Index1, std::size_t Ngon1>
auto boolean(mesh_wrapper<Index0, RealT, Ngon0, Dims> &form_wrapper0,
             mesh_wrapper<Index1, RealT, Ngon1, Dims> &form_wrapper1,
             tf::boolean_op op, const std::vector<int> &sheets) {
  auto graph = detail::boolean_graph(form_wrapper0, form_wrapper1, sheets);
  auto [result_mesh, tag_labels, face_labels] = tf::make_csg_mesh<RealT>(
      graph, tf::csg::make_boolean_expr(op), tf::return_source_ids);
  auto labels = tf::csg::graph::make_boolean_labels(tag_labels);
  // Extract mesh as (faces, points) - move ownership
  return nanobind::make_tuple(make_numpy_array(std::move(result_mesh)),
                              make_numpy_array(std::move(labels)),
                              make_numpy_array(std::move(face_labels)));
}

template <typename Index0, typename RealT, std::size_t Ngon0, std::size_t Dims,
          typename Index1, std::size_t Ngon1>
auto boolean(mesh_wrapper<Index0, RealT, Ngon0, Dims> &form_wrapper0,
             mesh_wrapper<Index1, RealT, Ngon1, Dims> &form_wrapper1,
             tf::boolean_op op, tf::return_curves_t,
             const std::vector<int> &sheets) {
  auto graph = detail::boolean_graph(form_wrapper0, form_wrapper1, sheets);
  auto [result_mesh, tag_labels, face_labels] = tf::make_csg_mesh<RealT>(
      graph, tf::csg::make_boolean_expr(op), tf::return_source_ids);
  auto labels = tf::csg::graph::make_boolean_labels(tag_labels);
  auto curves = tf::make_intersection_curves<RealT>(graph);

  // Extract mesh as (faces, points) - move ownership
  auto mesh_pair = make_numpy_array(std::move(result_mesh));

  // Extract labels buffer - move ownership
  auto labels_array = make_numpy_array(std::move(labels));
  auto face_labels_array = make_numpy_array(std::move(face_labels));

  // Extract curves as ((paths_offsets, paths_data), curve_points) - move
  // ownership
  auto [paths, c_points] = make_numpy_array(std::move(curves));
  auto curve_pair = nanobind::make_tuple(
      nanobind::make_tuple(paths.first, paths.second), std::move(c_points));
  return nanobind::make_tuple(std::move(mesh_pair), std::move(labels_array),
                              std::move(face_labels_array),
                              std::move(curve_pair));
}

// Helper function to convert Python int to C++ boolean_op enum
inline tf::boolean_op int_to_boolean_op(int op) {
  switch (op) {
  case 0:
    return tf::boolean_op::merge; // union
  case 1:
    return tf::boolean_op::intersection;
  case 2:
    return tf::boolean_op::left_difference; // difference
  case 3:
    // the facade swaps operands to reach an implemented index-type
    // ordering, which turns a difference around
    return tf::boolean_op::right_difference;
  default:
    throw std::invalid_argument("Invalid boolean operation: must be 0 (union), "
                                "1 (intersection), 2 (difference), or 3 "
                                "(reversed difference)");
  }
}

} // namespace tf::py

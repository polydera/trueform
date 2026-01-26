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
#include "../spatial/mesh.hpp"
#include "../util/make_numpy_array.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <trueform/cut/make_boolean.hpp>

namespace tf::py {
template <typename Index0, typename RealT, std::size_t Ngon0, std::size_t Dims,
          typename Index1, std::size_t Ngon1>
auto boolean(mesh_wrapper<Index0, RealT, Ngon0, Dims> &form_wrapper0,
             mesh_wrapper<Index1, RealT, Ngon1, Dims> &form_wrapper1,
             tf::boolean_op op) {
  bool has0 = form_wrapper0.has_transformation();
  bool has1 = form_wrapper1.has_transformation();
  auto form0 = form_wrapper0.make_primitive_range() |
               tf::tag(form_wrapper0.manifold_edge_link()) |
               tf::tag(form_wrapper0.face_membership()) |
               tf::tag(form_wrapper0.tree());
  auto form1 = form_wrapper1.make_primitive_range() |
               tf::tag(form_wrapper1.manifold_edge_link()) |
               tf::tag(form_wrapper1.face_membership()) |
               tf::tag(form_wrapper1.tree());
  auto make_return = [&op](auto &&form0, auto form1) {
    auto [result_mesh, labels] = tf::make_boolean(form0, form1, op);
    // Extract mesh as (faces, points) - move ownership
    return nanobind::make_tuple(make_numpy_array(std::move(result_mesh)),
                                make_numpy_array(std::move(labels)));
  };
  if (has0 && has1)
    return make_return(
        form0 | tf::tag(tf::make_frame(form_wrapper0.transformation_view())),
        form1 | tf::tag(tf::make_frame(form_wrapper1.transformation_view())));
  else if (has0 && !has1)
    return make_return(
        form0 | tf::tag(tf::make_frame(form_wrapper0.transformation_view())),
        form1);
  else if (!has0 && has1)
    return make_return(
        form0,
        form1 | tf::tag(tf::make_frame(form_wrapper1.transformation_view())));
  else
    return make_return(form0, form1);
}

template <typename Index0, typename RealT, std::size_t Ngon0, std::size_t Dims,
          typename Index1, std::size_t Ngon1>
auto boolean(mesh_wrapper<Index0, RealT, Ngon0, Dims> &form_wrapper0,
             mesh_wrapper<Index1, RealT, Ngon1, Dims> &form_wrapper1,
             tf::boolean_op op, tf::return_curves_t) {
  bool has0 = form_wrapper0.has_transformation();
  bool has1 = form_wrapper1.has_transformation();
  auto form0 = form_wrapper0.make_primitive_range() |
               tf::tag(form_wrapper0.manifold_edge_link()) |
               tf::tag(form_wrapper0.face_membership()) |
               tf::tag(form_wrapper0.tree());
  auto form1 = form_wrapper1.make_primitive_range() |
               tf::tag(form_wrapper1.manifold_edge_link()) |
               tf::tag(form_wrapper1.face_membership()) |
               tf::tag(form_wrapper1.tree());
  auto make_return = [&op](auto &&form0, auto form1) {
    auto [result_mesh, labels, curves] =
        tf::make_boolean(form0, form1, op, tf::return_curves);
    // Extract mesh as (faces, points) - move ownership
    auto mesh_pair = make_numpy_array(std::move(result_mesh));

    // Extract labels buffer - move ownership
    auto labels_array = make_numpy_array(std::move(labels));

    // Extract curves as ((paths_offsets, paths_data), curve_points) - move
    // ownership
    auto [paths, c_points] = make_numpy_array(std::move(curves));
    auto curve_pair = nanobind::make_tuple(
        nanobind::make_tuple(paths.first, paths.second), std::move(c_points));
    return nanobind::make_tuple(std::move(mesh_pair), std::move(labels_array),
                                std::move(curve_pair));
  };
  if (has0 && has1)
    return make_return(
        form0 | tf::tag(tf::make_frame(form_wrapper0.transformation_view())),
        form1 | tf::tag(tf::make_frame(form_wrapper1.transformation_view())));
  else if (has0 && !has1)
    return make_return(
        form0 | tf::tag(tf::make_frame(form_wrapper0.transformation_view())),
        form1);
  else if (!has0 && has1)
    return make_return(
        form0,
        form1 | tf::tag(tf::make_frame(form_wrapper1.transformation_view())));
  else
    return make_return(form0, form1);
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
  default:
    throw std::invalid_argument("Invalid boolean operation: must be 0 (union), "
                                "1 (intersection), or 2 (difference)");
  }
}

} // namespace tf::py

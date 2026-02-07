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
#include <trueform/cut/embedded_intersection_curves.hpp>

namespace tf::py {
template <typename Index0, typename RealT, std::size_t Ngon0, std::size_t Dims,
          typename Index1, std::size_t Ngon1>
auto embedded_intersection_curves(
    mesh_wrapper<Index0, RealT, Ngon0, Dims> &form_wrapper0,
    mesh_wrapper<Index1, RealT, Ngon1, Dims> &form_wrapper1) {
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
  auto make_return = [](auto &&form0, auto form1) {
    auto result_mesh = tf::embedded_intersection_curves(form0, form1);
    return make_numpy_array(std::move(result_mesh));
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
auto embedded_intersection_curves(
    mesh_wrapper<Index0, RealT, Ngon0, Dims> &form_wrapper0,
    mesh_wrapper<Index1, RealT, Ngon1, Dims> &form_wrapper1,
    tf::return_curves_t) {
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
  auto make_return = [](auto &&form0, auto form1) {
    auto [result_mesh, curves] =
        tf::embedded_intersection_curves(form0, form1, tf::return_curves);
    auto mesh_pair = make_numpy_array(std::move(result_mesh));
    auto [paths, c_points] = make_numpy_array(std::move(curves));
    auto curve_pair = nanobind::make_tuple(
        nanobind::make_tuple(paths.first, paths.second), std::move(c_points));
    return nanobind::make_tuple(std::move(mesh_pair), std::move(curve_pair));
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

} // namespace tf::py

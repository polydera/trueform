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
#include <nanobind/stl/vector.h>
#include <trueform/core/transformation.hpp>
#include <trueform/intersect/make_intersection_curves.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include <trueform/topology/policy/manifold_edge_link.hpp>

namespace tf::py {

template <typename Index0, typename RealT, std::size_t Ngon0, std::size_t Dims,
          typename Index1, std::size_t Ngon1>
auto intersection_curves(
    mesh_wrapper<Index0, RealT, Ngon0, Dims> &form_wrapper0,
    mesh_wrapper<Index1, RealT, Ngon1, Dims> &form_wrapper1,
    tf::intersect_mode mode = tf::intersect_mode::sos) {
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
  auto make_return = [mode](auto &&form0, auto &&form1) {
    auto curves = tf::make_intersection_curves(form0, form1, mode);
    auto [paths, c_points] = make_numpy_array(std::move(curves));
    return nanobind::make_tuple(nanobind::make_tuple(paths.first, paths.second),
                                std::move(c_points));
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

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto intersection_curves(
    std::vector<mesh_wrapper<Index, RealT, Ngon, Dims>> &wrappers,
    tf::intersect_mode mode = tf::intersect_mode::sos) {
  bool any_transformed = false;
  for (auto &w : wrappers)
    if (w.has_transformation())
      any_transformed = true;

  auto run = [mode](const auto &forms) {
    auto curves = tf::make_intersection_curves(forms, mode);
    auto [paths, c_points] = make_numpy_array(std::move(curves));
    return nanobind::make_tuple(nanobind::make_tuple(paths.first, paths.second),
                                std::move(c_points));
  };

  auto range = tf::make_range(wrappers.data(), wrappers.size());

  if (any_transformed) {
    auto identity = tf::make_identity_transformation<RealT, Dims>();
    auto tagged = tf::make_mapped_range(range, [&](auto &w) {
      auto fm = w.face_membership();
      auto mel = w.manifold_edge_link();
      tf::transformation<RealT, Dims> tv = w.has_transformation()
          ? tf::transformation<RealT, Dims>(w.transformation_view())
          : identity;
      return w.make_primitive_range() | tf::tag(w.tree()) | tf::tag(fm) |
             tf::tag(mel) | tf::tag(tv);
    });
    return run(tagged);
  }

  auto tagged = tf::make_mapped_range(range, [](auto &w) {
    auto fm = w.face_membership();
    auto mel = w.manifold_edge_link();
    return w.make_primitive_range() | tf::tag(w.tree()) | tf::tag(fm) |
           tf::tag(mel);
  });
  return run(tagged);
}

} // namespace tf::py

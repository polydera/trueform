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
#include <trueform/core/angle.hpp>
#include <trueform/remesh/isotropic_remeshed.hpp>
#include <trueform/core.hpp>

namespace tf::py {

template <typename Index, typename RealT>
auto isotropic_remeshed_impl(mesh_wrapper<Index, RealT, 3, 3> &wrapper,
                             RealT target_length, int iterations,
                             int relaxation_iters, RealT max_aspect_ratio,
                             RealT lambda, bool preserve_boundary,
                             bool use_quadric, bool parallel,
                             double feature_angle, RealT feature_weight) {
  auto polys = wrapper.make_primitive_range();
  auto fm = wrapper.face_membership();
  tf::half_edges<Index> he;
  he.build(polys.faces(), fm);
  auto tagged = polys | tf::tag(fm) | tf::tag(he);

  tf::remesh_config<RealT> config{
      target_length,  iterations,        relaxation_iters,
      max_aspect_ratio, lambda,          preserve_boundary,
      use_quadric,    parallel,          tf::rad<RealT>(RealT(feature_angle)),
      feature_weight};

  auto run = [&](auto &&form) {
    auto [result, result_he] = tf::isotropic_remeshed(form, config);
    return make_numpy_array(std::move(result));
  };

  if (wrapper.has_transformation())
    return run(
        tagged |
        tf::tag(wrapper.transformation_view()));
  else
    return run(tagged);
}

} // namespace tf::py

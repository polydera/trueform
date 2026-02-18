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
#include <trueform/remesh/isotropic_remeshed.hpp>
#include <trueform/core.hpp>

namespace tf::py {

template <typename Index, typename RealT>
auto isotropic_remeshed_impl(mesh_wrapper<Index, RealT, 3, 3> &wrapper,
                             RealT target_length, int iterations,
                             int relaxation_iters, RealT max_aspect_ratio,
                             RealT lambda, bool preserve_boundary,
                             bool use_quadric, bool parallel) {
  auto polys = wrapper.make_primitive_range();
  auto fm = wrapper.face_membership();
  tf::half_edges<Index> he;
  he.build(polys.faces(), fm);
  auto tagged = polys | tf::tag(fm) | tf::tag(he);

  tf::remesh_config<RealT> config;
  config.target_length = target_length;
  config.iterations = iterations;
  config.relaxation_iters = relaxation_iters;
  config.max_aspect_ratio = max_aspect_ratio;
  config.lambda = lambda;
  config.preserve_boundary = preserve_boundary;
  config.use_quadric = use_quadric;
  config.parallel = parallel;

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

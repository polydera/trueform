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
#include <optional>
#include <trueform/core/angle.hpp>
#include <trueform/remesh/simplified.hpp>
#include <trueform/core.hpp>

namespace tf::py {

template <typename Index, typename RealT>
auto simplified_impl(
    mesh_wrapper<Index, RealT, 3, 3> &wrapper, RealT error_rel,
    int optimize_iterations, RealT min_quality, bool preserve_boundary,
    double stabilizer, bool parallel, double feature_angle, RealT feature_weight,
    int iterations, int relaxation_iters, RealT lambda,
    std::optional<
        nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>>
        regions) {
  auto polys = wrapper.make_primitive_range();
  auto fm = wrapper.face_membership();
  tf::half_edges<Index> he;
  he.build(polys.faces(), fm);
  auto tagged = polys | tf::tag(fm) | tf::tag(he);

  tf::simplify_config<RealT> config;
  config.error_rel = error_rel;
  config.optimize_iterations = optimize_iterations;
  config.iterations = iterations;
  config.relaxation_iters = relaxation_iters;
  config.lambda = lambda;
  config.min_quality = min_quality;
  config.preserve_boundary = preserve_boundary;
  config.stabilizer = stabilizer;
  config.parallel = parallel;
  config.feature_angle = tf::rad<RealT>(RealT(feature_angle));
  config.feature_weight = feature_weight;

  auto run = [&](auto &&form) -> nanobind::object {
    if (regions) {
      auto r = tf::make_range(static_cast<const int *>(regions->data()),
                              regions->size());
      auto [result, result_he, labels] =
          tf::simplified(form, config, tf::preserve_regions(r));
      (void)result_he;
      auto fp = make_numpy_array(std::move(result));
      return nanobind::make_tuple(fp.first, fp.second,
                                  make_numpy_array(std::move(labels)));
    }
    auto [result, result_he] = tf::simplified(form, config);
    (void)result_he;
    auto fp = make_numpy_array(std::move(result));
    return nanobind::make_tuple(fp.first, fp.second);
  };

  if (wrapper.has_transformation())
    return run(
        tagged |
        tf::tag(wrapper.transformation_view()));
  else
    return run(tagged);
}

} // namespace tf::py

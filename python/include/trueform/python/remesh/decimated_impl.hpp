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
#include <trueform/remesh/decimated.hpp>
#include <trueform/core.hpp>

namespace tf::py {

template <typename Index, typename RealT>
auto decimated_impl(mesh_wrapper<Index, RealT, 3, 3> &wrapper,
                    RealT target_proportion, RealT max_aspect_ratio,
                    bool preserve_boundary, double stabilizer, bool parallel,
                    double feature_angle, RealT feature_weight) {
  auto polys = wrapper.make_primitive_range();
  auto fm = wrapper.face_membership();
  tf::half_edges<Index> he;
  he.build(polys.faces(), fm);
  auto tagged = polys | tf::tag(fm) | tf::tag(he);

  tf::decimate_config<RealT> config;
  config.max_aspect_ratio = max_aspect_ratio;
  config.preserve_boundary = preserve_boundary;
  config.stabilizer = stabilizer;
  config.parallel = parallel;
  config.feature_angle = tf::rad<RealT>(RealT(feature_angle));
  config.feature_weight = feature_weight;

  auto run = [&](auto &&form) {
    auto [result, result_he] = tf::decimated(form, target_proportion, config);
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

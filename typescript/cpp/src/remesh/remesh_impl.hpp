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

#include "trueform/core/angle.hpp"
#include "trueform/remesh/decimated.hpp"
#include "trueform/remesh/isotropic_remeshed.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"

namespace tf {
namespace ts {

// -- Sync --

template <typename Real>
auto sync_decimated(wasm_mesh<Real> &m, Real target_proportion,
                    Real max_aspect_ratio, bool preserve_boundary,
                    double stabilizer, bool parallel,
                    Real feature_angle, Real feature_weight)
    -> wasm_mesh<Real> {
  auto polys = m.polygons_range();
  auto fm = m.face_membership_range();
  auto &he = m.half_edges();
  auto tagged = polys | tf::tag(fm) | tf::tag(he);

  tf::decimate_config<Real> config;
  config.max_aspect_ratio = max_aspect_ratio;
  config.preserve_boundary = preserve_boundary;
  config.stabilizer = stabilizer;
  config.parallel = parallel;
  config.feature_angle = tf::rad<Real>(feature_angle);
  config.feature_weight = feature_weight;

  auto run = [&](auto &&form) -> wasm_mesh<Real> {
    auto [result, result_he] =
        tf::decimated(form, target_proportion, config);
    auto mesh = wasm_mesh<Real>::from_polygons_buffer(std::move(result));
    mesh.set_half_edges(std::move(result_he));
    return mesh;
  };

  if (m.has_transformation())
    return run(tagged | tf::tag(m.transformation_view()));
  else
    return run(tagged);
}

template <typename Real>
auto sync_isotropic_remeshed(wasm_mesh<Real> &m, Real target_length,
                             int iterations, int relaxation_iters,
                             Real max_aspect_ratio, Real lambda,
                             bool preserve_boundary, bool use_quadric,
                             bool parallel, Real feature_angle,
                             Real feature_weight) -> wasm_mesh<Real> {
  auto polys = m.polygons_range();
  auto fm = m.face_membership_range();
  auto &he = m.half_edges();
  auto tagged = polys | tf::tag(fm) | tf::tag(he);

  tf::remesh_config<Real> config{target_length, iterations, relaxation_iters,
                                 max_aspect_ratio, lambda, preserve_boundary,
                                 use_quadric, parallel,
                                 tf::rad<Real>(feature_angle),
                                 feature_weight};

  auto run = [&](auto &&form) -> wasm_mesh<Real> {
    auto [result, result_he] = tf::isotropic_remeshed(form, config);
    auto mesh = wasm_mesh<Real>::from_polygons_buffer(std::move(result));
    mesh.set_half_edges(std::move(result_he));
    return mesh;
  };

  if (m.has_transformation())
    return run(tagged | tf::tag(m.transformation_view()));
  else
    return run(tagged);
}

// -- Async --

template <typename Real>
auto async_decimated(wasm_mesh<Real> &m, Real target_proportion,
                     Real max_aspect_ratio, bool preserve_boundary,
                     double stabilizer, bool parallel,
                     Real feature_angle, Real feature_weight) -> promise_t {
  return promise([a = m, target_proportion, max_aspect_ratio,
                  preserve_boundary, stabilizer, parallel, feature_angle,
                  feature_weight]() -> wasm_mesh<Real> {
    return sync_decimated<Real>(const_cast<wasm_mesh<Real> &>(a),
                                target_proportion, max_aspect_ratio,
                                preserve_boundary, stabilizer, parallel,
                                feature_angle, feature_weight);
  });
}

template <typename Real>
auto async_isotropic_remeshed(wasm_mesh<Real> &m, Real target_length,
                              int iterations, int relaxation_iters,
                              Real max_aspect_ratio, Real lambda,
                              bool preserve_boundary, bool use_quadric,
                              bool parallel, Real feature_angle,
                              Real feature_weight) -> promise_t {
  return promise([a = m, target_length, iterations, relaxation_iters,
                  max_aspect_ratio, lambda, preserve_boundary, use_quadric,
                  parallel, feature_angle, feature_weight]()
                     -> wasm_mesh<Real> {
    return sync_isotropic_remeshed<Real>(
        const_cast<wasm_mesh<Real> &>(a), target_length, iterations,
        relaxation_iters, max_aspect_ratio, lambda, preserve_boundary,
        use_quadric, parallel, feature_angle, feature_weight);
  });
}

} // namespace ts
} // namespace tf

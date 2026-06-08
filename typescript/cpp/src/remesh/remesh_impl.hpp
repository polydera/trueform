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
#include "trueform/remesh/simplified.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"

namespace tf {
namespace ts {

// Result of a region-preserving remesh: the new mesh plus the output mesh's
// per-face region labels. When the caller passes no regions (an empty labels
// array) the library runs the plain path and `regions` comes back empty; the
// TS wrapper then returns just the mesh.
template <typename Real> struct remesh_result_t {
  wasm_mesh<Real> mesh;
  wasm_ndarray<int> regions;
};

// -- Sync --

template <typename Real>
auto sync_decimated(wasm_mesh<Real> &m, Real target_proportion,
                    Real min_quality, bool preserve_boundary,
                    double stabilizer, bool parallel,
                    Real feature_angle, Real feature_weight,
                    wasm_ndarray<int> regions)
    -> remesh_result_t<Real> {
  auto polys = m.polygons_range();
  auto fm = m.face_membership_range();
  auto &he = m.half_edges();
  auto tagged = polys | tf::tag(fm) | tf::tag(he);

  tf::decimate_config<Real> config;
  config.min_quality = min_quality;
  config.preserve_boundary = preserve_boundary;
  config.stabilizer = stabilizer;
  config.parallel = parallel;
  config.feature_angle = tf::rad<Real>(feature_angle);
  config.feature_weight = feature_weight;

  auto run = [&](auto &&form) -> remesh_result_t<Real> {
    auto [result, result_he, labels] = tf::decimated(
        form, target_proportion, config,
        tf::preserve_regions(regions.make_range()));
    remesh_result_t<Real> out;
    out.mesh = wasm_mesh<Real>::from_polygons_buffer(std::move(result));
    out.mesh.set_half_edges(std::move(result_he));
    int n = int(labels.size());
    out.regions = wasm_ndarray<int>::from_buffer(std::move(labels), {n});
    return out;
  };

  if (m.has_transformation())
    return run(tagged | tf::tag(m.transformation_view()));
  else
    return run(tagged);
}

template <typename Real>
auto sync_isotropic_remeshed(wasm_mesh<Real> &m, Real target_length,
                             int iterations, int relaxation_iters,
                             Real min_quality, Real lambda,
                             bool preserve_boundary, bool use_quadric,
                             bool parallel, Real feature_angle,
                             Real feature_weight, wasm_ndarray<int> regions)
    -> remesh_result_t<Real> {
  auto polys = m.polygons_range();
  auto fm = m.face_membership_range();
  auto &he = m.half_edges();
  auto tagged = polys | tf::tag(fm) | tf::tag(he);

  tf::isotropic_remesh_config<Real> config{target_length, iterations, relaxation_iters,
                                 min_quality, lambda, preserve_boundary,
                                 use_quadric, parallel,
                                 tf::rad<Real>(feature_angle),
                                 feature_weight};

  auto run = [&](auto &&form) -> remesh_result_t<Real> {
    auto [result, result_he, labels] = tf::isotropic_remeshed(
        form, config, tf::preserve_regions(regions.make_range()));
    remesh_result_t<Real> out;
    out.mesh = wasm_mesh<Real>::from_polygons_buffer(std::move(result));
    out.mesh.set_half_edges(std::move(result_he));
    int n = int(labels.size());
    out.regions = wasm_ndarray<int>::from_buffer(std::move(labels), {n});
    return out;
  };

  if (m.has_transformation())
    return run(tagged | tf::tag(m.transformation_view()));
  else
    return run(tagged);
}

template <typename Real>
auto sync_simplified(wasm_mesh<Real> &m, Real error_rel,
                     int optimize_iterations, Real min_quality,
                     bool preserve_boundary, double stabilizer, bool parallel,
                     Real feature_angle, Real feature_weight, int iterations,
                     int relaxation_iters, Real lambda,
                     wasm_ndarray<int> regions) -> remesh_result_t<Real> {
  auto polys = m.polygons_range();
  auto fm = m.face_membership_range();
  auto &he = m.half_edges();
  auto tagged = polys | tf::tag(fm) | tf::tag(he);

  tf::simplify_config<Real> config;
  config.error_rel = error_rel;
  config.optimize_iterations = optimize_iterations;
  config.iterations = iterations;
  config.relaxation_iters = relaxation_iters;
  config.lambda = lambda;
  config.min_quality = min_quality;
  config.preserve_boundary = preserve_boundary;
  config.stabilizer = stabilizer;
  config.parallel = parallel;
  config.feature_angle = tf::rad<Real>(feature_angle);
  config.feature_weight = feature_weight;

  auto run = [&](auto &&form) -> remesh_result_t<Real> {
    auto [result, result_he, labels] = tf::simplified(
        form, config, tf::preserve_regions(regions.make_range()));
    remesh_result_t<Real> out;
    out.mesh = wasm_mesh<Real>::from_polygons_buffer(std::move(result));
    out.mesh.set_half_edges(std::move(result_he));
    int n = int(labels.size());
    out.regions = wasm_ndarray<int>::from_buffer(std::move(labels), {n});
    return out;
  };

  if (m.has_transformation())
    return run(tagged | tf::tag(m.transformation_view()));
  else
    return run(tagged);
}

// -- Async --

template <typename Real>
auto async_decimated(wasm_mesh<Real> &m, Real target_proportion,
                     Real min_quality, bool preserve_boundary,
                     double stabilizer, bool parallel,
                     Real feature_angle, Real feature_weight,
                     wasm_ndarray<int> regions) -> promise_t {
  return promise([a = m, target_proportion, min_quality,
                  preserve_boundary, stabilizer, parallel, feature_angle,
                  feature_weight, regions]() -> remesh_result_t<Real> {
    return sync_decimated<Real>(const_cast<wasm_mesh<Real> &>(a),
                                target_proportion, min_quality,
                                preserve_boundary, stabilizer, parallel,
                                feature_angle, feature_weight, regions);
  });
}

template <typename Real>
auto async_isotropic_remeshed(wasm_mesh<Real> &m, Real target_length,
                              int iterations, int relaxation_iters,
                              Real min_quality, Real lambda,
                              bool preserve_boundary, bool use_quadric,
                              bool parallel, Real feature_angle,
                              Real feature_weight,
                              wasm_ndarray<int> regions) -> promise_t {
  return promise([a = m, target_length, iterations, relaxation_iters,
                  min_quality, lambda, preserve_boundary, use_quadric,
                  parallel, feature_angle, feature_weight, regions]()
                     -> remesh_result_t<Real> {
    return sync_isotropic_remeshed<Real>(
        const_cast<wasm_mesh<Real> &>(a), target_length, iterations,
        relaxation_iters, min_quality, lambda, preserve_boundary,
        use_quadric, parallel, feature_angle, feature_weight, regions);
  });
}

template <typename Real>
auto async_simplified(wasm_mesh<Real> &m, Real error_rel,
                      int optimize_iterations, Real min_quality,
                      bool preserve_boundary, double stabilizer, bool parallel,
                      Real feature_angle, Real feature_weight, int iterations,
                      int relaxation_iters, Real lambda,
                      wasm_ndarray<int> regions) -> promise_t {
  return promise([a = m, error_rel, optimize_iterations, min_quality,
                  preserve_boundary, stabilizer, parallel, feature_angle,
                  feature_weight, iterations, relaxation_iters, lambda,
                  regions]() -> remesh_result_t<Real> {
    return sync_simplified<Real>(const_cast<wasm_mesh<Real> &>(a), error_rel,
                                 optimize_iterations, min_quality,
                                 preserve_boundary, stabilizer, parallel,
                                 feature_angle, feature_weight, iterations,
                                 relaxation_iters, lambda, regions);
  });
}

} // namespace ts
} // namespace tf

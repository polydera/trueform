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

#include "trueform/core/angle.hpp"
#include "trueform/remesh/decimated.hpp"
#include "trueform/remesh/isotropic_remeshed.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// -- Sync --

auto sync_decimated(wasm_mesh &m, float target_proportion,
                    float max_aspect_ratio, bool preserve_boundary,
                    double stabilizer, bool parallel,
                    float feature_angle, float feature_weight) -> wasm_mesh {
  auto polys = m.polygons_range();
  auto fm = m.face_membership_range();
  auto &he = m.half_edges();
  auto tagged = polys | tf::tag(fm) | tf::tag(he);

  tf::decimate_config<float> config;
  config.max_aspect_ratio = max_aspect_ratio;
  config.preserve_boundary = preserve_boundary;
  config.stabilizer = stabilizer;
  config.parallel = parallel;
  config.feature_angle = tf::rad<float>(feature_angle);
  config.feature_weight = feature_weight;

  auto run = [&](auto &&form) -> wasm_mesh {
    auto [result, result_he] =
        tf::decimated(form, target_proportion, config);
    auto mesh = wasm_mesh::from_polygons_buffer(std::move(result));
    mesh.set_half_edges(std::move(result_he));
    return mesh;
  };

  if (m.has_transformation())
    return run(tagged | tf::tag(m.transformation_view()));
  else
    return run(tagged);
}

auto sync_isotropic_remeshed(wasm_mesh &m, float target_length,
                              int iterations, int relaxation_iters,
                              float max_aspect_ratio, float lambda,
                              bool preserve_boundary, bool use_quadric,
                              bool parallel, float feature_angle,
                              float feature_weight) -> wasm_mesh {
  auto polys = m.polygons_range();
  auto fm = m.face_membership_range();
  auto &he = m.half_edges();
  auto tagged = polys | tf::tag(fm) | tf::tag(he);

  tf::remesh_config<float> config{target_length, iterations, relaxation_iters,
                                  max_aspect_ratio, lambda, preserve_boundary,
                                  use_quadric, parallel,
                                  tf::rad<float>(feature_angle),
                                  feature_weight};

  auto run = [&](auto &&form) -> wasm_mesh {
    auto [result, result_he] = tf::isotropic_remeshed(form, config);
    auto mesh = wasm_mesh::from_polygons_buffer(std::move(result));
    mesh.set_half_edges(std::move(result_he));
    return mesh;
  };

  if (m.has_transformation())
    return run(tagged | tf::tag(m.transformation_view()));
  else
    return run(tagged);
}

// -- Async --

auto async_decimated(wasm_mesh &m, float target_proportion,
                     float max_aspect_ratio, bool preserve_boundary,
                     double stabilizer, bool parallel,
                     float feature_angle, float feature_weight) -> promise_t {
  return promise([a = m, target_proportion, max_aspect_ratio,
                  preserve_boundary, stabilizer, parallel, feature_angle,
                  feature_weight]() -> wasm_mesh {
    return sync_decimated(const_cast<wasm_mesh &>(a), target_proportion,
                          max_aspect_ratio, preserve_boundary, stabilizer,
                          parallel, feature_angle, feature_weight);
  });
}

auto async_isotropic_remeshed(wasm_mesh &m, float target_length,
                               int iterations, int relaxation_iters,
                               float max_aspect_ratio, float lambda,
                               bool preserve_boundary, bool use_quadric,
                               bool parallel, float feature_angle,
                               float feature_weight) -> promise_t {
  return promise([a = m, target_length, iterations, relaxation_iters,
                  max_aspect_ratio, lambda, preserve_boundary, use_quadric,
                  parallel, feature_angle, feature_weight]() -> wasm_mesh {
    return sync_isotropic_remeshed(
        const_cast<wasm_mesh &>(a), target_length, iterations,
        relaxation_iters, max_aspect_ratio, lambda, preserve_boundary,
        use_quadric, parallel, feature_angle, feature_weight);
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_remesh) {
  emscripten::function("decimated", &sync_decimated);
  emscripten::function("isotropic_remeshed", &sync_isotropic_remeshed);

  emscripten::function("dispatch_decimated", &async_decimated);
  emscripten::function("dispatch_isotropic_remeshed",
                       &async_isotropic_remeshed);
}

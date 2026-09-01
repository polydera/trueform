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

#include "trueform/arrangement/make_mesh_arrangements.hpp"
#include "trueform/topology/triangulation_type.hpp"
#include "trueform/ts/core/build_intersect_structures.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_curves.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <emscripten/val.h>
#include <cstdint>
#include <vector>

namespace tf {
namespace ts {

// ============================================================================
// Result types — the merged arrangement mesh + (optional) intersection curves
// carry the inputs' coordinate dtype; tag / face labels stay int32.
// ============================================================================

template <typename Real> struct arrangement_result_t {
  wasm_mesh<Real> mesh;
  wasm_ndarray<std::int32_t> tag_labels;
  wasm_ndarray<std::int32_t> face_labels;
};

template <typename Real> struct arrangement_result_with_curves_t {
  wasm_mesh<Real> mesh;
  wasm_ndarray<std::int32_t> tag_labels;
  wasm_ndarray<std::int32_t> face_labels;
  wasm_curves<Real> curves;
};

// ============================================================================
// Extraction — pulled out of the JS array on the main thread before any
// async dispatch, since `emscripten::val` can't cross threads.
// ============================================================================

template <typename Real>
auto extract_meshes(emscripten::val js_meshes) -> std::vector<wasm_mesh<Real>> {
  auto n = js_meshes["length"].as<int>();
  if (n < 2)
    throw std::runtime_error("meshArrangement: need at least 2 meshes");
  std::vector<wasm_mesh<Real>> meshes;
  meshes.reserve(n);
  for (int i = 0; i < n; ++i)
    meshes.push_back(
        js_meshes[i].as<wasm_mesh<Real>>(emscripten::allow_raw_pointers()));
  return meshes;
}

// ============================================================================
// Core drivers — stitch per-mesh transformations into the tagged form-range
// and call into the trueform arrangement pipeline.
// ============================================================================

template <typename Real, typename MeshRange>
auto run_arrangement(MeshRange &meshes, int mode, double tolerance,
                     int triangulation) -> arrangement_result_t<Real> {
  build_intersect_structures_all(meshes);
  auto mesh_range = tf::make_range(meshes);
  bool any_transformed = false;
  for (auto &m : meshes)
    if (m.has_transformation())
      any_transformed = true;

  auto run = [mode, tolerance,
              triangulation](const auto &forms) -> arrangement_result_t<Real> {
    auto [mesh, tag_labels, face_labels] = tf::make_mesh_arrangements(
        forms,
        tf::arrangement_config{
            tf::intersect_config{static_cast<tf::intersect_mode>(mode),
                                 tolerance},
            static_cast<tf::triangulation_type>(triangulation)});
    return {wasm_mesh<Real>::from_polygons_buffer(std::move(mesh)),
            wasm_ndarray<std::int32_t>::from_buffer(std::move(tag_labels)),
            wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels))};
  };

  if (any_transformed) {
    auto identity = tf::make_identity_transformation<Real, 3>();
    auto tagged = tf::make_mapped_range(mesh_range, [&](auto &m) {
      auto fm = m.face_membership_range();
      auto mel = m.manifold_edge_link_range();
      auto tv = m.has_transformation()
                    ? m.transformation_view()
                    : tf::make_transformation_view<3>(&identity(0, 0));
      return m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) |
             tf::tag(mel) | tf::tag(tv);
    });
    return run(tagged);
  }

  auto tagged = tf::make_mapped_range(mesh_range, [](auto &m) {
    auto fm = m.face_membership_range();
    auto mel = m.manifold_edge_link_range();
    return m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) |
           tf::tag(mel);
  });
  return run(tagged);
}

template <typename Real, typename MeshRange>
auto run_arrangement_with_curves(MeshRange &meshes, int mode, double tolerance,
                                 int triangulation)
    -> arrangement_result_with_curves_t<Real> {
  build_intersect_structures_all(meshes);
  auto mesh_range = tf::make_range(meshes);
  bool any_transformed = false;
  for (auto &m : meshes)
    if (m.has_transformation())
      any_transformed = true;

  auto run =
      [mode, tolerance, triangulation](
          const auto &forms) -> arrangement_result_with_curves_t<Real> {
    auto [mesh, tag_labels, face_labels, curves] =
        tf::make_mesh_arrangements(
            forms,
            tf::arrangement_config{
                tf::intersect_config{static_cast<tf::intersect_mode>(mode),
                                     tolerance},
                static_cast<tf::triangulation_type>(triangulation)},
            tf::return_curves);
    return {wasm_mesh<Real>::from_polygons_buffer(std::move(mesh)),
            wasm_ndarray<std::int32_t>::from_buffer(std::move(tag_labels)),
            wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels)),
            wasm_curves<Real>::from_curves_buffer(std::move(curves))};
  };

  if (any_transformed) {
    auto identity = tf::make_identity_transformation<Real, 3>();
    auto tagged = tf::make_mapped_range(mesh_range, [&](auto &m) {
      auto fm = m.face_membership_range();
      auto mel = m.manifold_edge_link_range();
      auto tv = m.has_transformation()
                    ? m.transformation_view()
                    : tf::make_transformation_view<3>(&identity(0, 0));
      return m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) |
             tf::tag(mel) | tf::tag(tv);
    });
    return run(tagged);
  }

  auto tagged = tf::make_mapped_range(mesh_range, [](auto &m) {
    auto fm = m.face_membership_range();
    auto mel = m.manifold_edge_link_range();
    return m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) |
           tf::tag(mel);
  });
  return run(tagged);
}

// ============================================================================
// Sync entry points — extract the JS array on the main thread, then run.
// ============================================================================

template <typename Real>
auto sync_mesh_arrangement(emscripten::val js_meshes, int mode, double tolerance,
                          int triangulation) -> arrangement_result_t<Real> {
  auto meshes = extract_meshes<Real>(js_meshes);
  return run_arrangement<Real>(meshes, mode, tolerance, triangulation);
}

template <typename Real>
auto sync_mesh_arrangement_with_curves(emscripten::val js_meshes, int mode,
                                       double tolerance, int triangulation)
    -> arrangement_result_with_curves_t<Real> {
  auto meshes = extract_meshes<Real>(js_meshes);
  return run_arrangement_with_curves<Real>(meshes, mode, tolerance,
                                           triangulation);
}

// ============================================================================
// Async — extract the JS array on the main thread, then capture the
// std::vector<wasm_mesh<Real>> by move into the worker lambda.
// ============================================================================

template <typename Real>
auto async_mesh_arrangement(emscripten::val js_meshes, int mode, double tolerance,
                           int triangulation) -> promise_t {
  auto meshes = extract_meshes<Real>(js_meshes);
  return promise([ms = std::move(meshes), mode, tolerance,
                  triangulation]() -> arrangement_result_t<Real> {
    auto &meshes = const_cast<std::vector<wasm_mesh<Real>> &>(ms);
    return run_arrangement<Real>(meshes, mode, tolerance, triangulation);
  });
}

template <typename Real>
auto async_mesh_arrangement_with_curves(emscripten::val js_meshes, int mode,
                                        double tolerance, int triangulation)
    -> promise_t {
  auto meshes = extract_meshes<Real>(js_meshes);
  return promise([ms = std::move(meshes), mode, tolerance,
                  triangulation]() -> arrangement_result_with_curves_t<Real> {
    auto &meshes = const_cast<std::vector<wasm_mesh<Real>> &>(ms);
    return run_arrangement_with_curves<Real>(meshes, mode, tolerance,
                                             triangulation);
  });
}

} // namespace ts
} // namespace tf

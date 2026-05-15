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

#include "trueform/intersect/intersect_mode.hpp"
#include "trueform/intersect/make_intersection_curves.hpp"
#include "trueform/intersect/make_isocurves.hpp"
#include "trueform/intersect/make_self_intersection_curves.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_curves.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <emscripten/val.h>
#include <vector>

namespace tf {
namespace ts {

// ============================================================================
// Intersection curves (mesh0 x mesh1 -> curves)
// ============================================================================

template <typename Real>
auto sync_intersection_curves(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                              int mode) -> wasm_curves<Real> {
  bool has0 = m0.has_transformation();
  bool has1 = m1.has_transformation();
  auto fm0 = m0.face_membership_range();
  auto mel0 = m0.manifold_edge_link_range();
  auto fm1 = m1.face_membership_range();
  auto mel1 = m1.manifold_edge_link_range();
  auto make_return = [&](auto &&poly0, auto &&poly1) -> wasm_curves<Real> {
    auto cb = tf::make_intersection_curves(
        poly0 | tf::tag(m0.tree()) | tf::tag(fm0) | tf::tag(mel0),
        poly1 | tf::tag(m1.tree()) | tf::tag(fm1) | tf::tag(mel1),
        static_cast<tf::intersect_mode>(mode));
    return wasm_curves<Real>::from_curves_buffer(std::move(cb));
  };
  if (has0 && has1)
    return make_return(
        m0.polygons_range() | tf::tag(m0.transformation_view()),
        m1.polygons_range() | tf::tag(m1.transformation_view()));
  else if (has0)
    return make_return(
        m0.polygons_range() | tf::tag(m0.transformation_view()),
        m1.polygons_range());
  else if (has1)
    return make_return(m0.polygons_range(),
                       m1.polygons_range() | tf::tag(m1.transformation_view()));
  else
    return make_return(m0.polygons_range(), m1.polygons_range());
}

template <typename Real>
auto async_intersection_curves(wasm_mesh<Real> &m0, wasm_mesh<Real> &m1,
                               int mode) -> promise_t {
  return promise([a = m0, b = m1, mode]() -> wasm_curves<Real> {
    return sync_intersection_curves<Real>(
        const_cast<wasm_mesh<Real> &>(a),
        const_cast<wasm_mesh<Real> &>(b), mode);
  });
}

// ============================================================================
// Intersection curves list (N meshes x mode -> curves)
// ============================================================================

template <typename Real>
auto extract_meshes(emscripten::val js_meshes) -> std::vector<wasm_mesh<Real>> {
  auto n = js_meshes["length"].as<int>();
  std::vector<wasm_mesh<Real>> meshes;
  meshes.reserve(n);
  for (int i = 0; i < n; ++i)
    meshes.push_back(
        js_meshes[i].as<wasm_mesh<Real>>(emscripten::allow_raw_pointers()));
  return meshes;
}

template <typename Real>
auto intersection_curves_list_impl(std::vector<wasm_mesh<Real>> &meshes,
                                   int mode) -> wasm_curves<Real> {
  auto mesh_range = tf::make_range(meshes);
  bool any_transformed = false;
  for (auto &m : meshes)
    if (m.has_transformation())
      any_transformed = true;

  auto run = [mode](const auto &forms) -> wasm_curves<Real> {
    auto cb = tf::make_intersection_curves(
        forms, static_cast<tf::intersect_mode>(mode));
    return wasm_curves<Real>::from_curves_buffer(std::move(cb));
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
    return m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) | tf::tag(mel);
  });
  return run(tagged);
}

template <typename Real>
auto sync_intersection_curves_list(emscripten::val js_meshes, int mode)
    -> wasm_curves<Real> {
  auto meshes = extract_meshes<Real>(js_meshes);
  return intersection_curves_list_impl<Real>(meshes, mode);
}

template <typename Real>
auto async_intersection_curves_list(emscripten::val js_meshes, int mode)
    -> promise_t {
  auto meshes = extract_meshes<Real>(js_meshes);
  return promise([ms = std::move(meshes), mode]() -> wasm_curves<Real> {
    auto &meshes = const_cast<std::vector<wasm_mesh<Real>> &>(ms);
    return intersection_curves_list_impl<Real>(meshes, mode);
  });
}

// ============================================================================
// Self-intersection curves (mesh -> curves)
// ============================================================================

template <typename Real>
auto sync_self_intersection_curves(wasm_mesh<Real> &m, int mode)
    -> wasm_curves<Real> {
  auto fm = m.face_membership_range();
  auto mel = m.manifold_edge_link_range();
  auto cb = tf::make_self_intersection_curves(
      m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) | tf::tag(mel),
      static_cast<tf::intersect_mode>(mode));
  return wasm_curves<Real>::from_curves_buffer(std::move(cb));
}

template <typename Real>
auto async_self_intersection_curves(wasm_mesh<Real> &m, int mode)
    -> promise_t {
  return promise([a = m, mode]() -> wasm_curves<Real> {
    return sync_self_intersection_curves<Real>(
        const_cast<wasm_mesh<Real> &>(a), mode);
  });
}

// ============================================================================
// Isocontours — single threshold
// ============================================================================

template <typename Real>
auto sync_isocontours(wasm_mesh<Real> &mesh, wasm_ndarray<Real> &scalars,
                      Real cut_value) -> wasm_curves<Real> {
  auto cb = tf::make_isocontours(mesh.polygons_range(), scalars.make_range(),
                                 cut_value);
  return wasm_curves<Real>::from_curves_buffer(std::move(cb));
}

template <typename Real>
auto async_isocontours(wasm_mesh<Real> &mesh, wasm_ndarray<Real> &scalars,
                       Real cut_value) -> promise_t {
  return promise([m = mesh, s = scalars, cut_value]() -> wasm_curves<Real> {
    return sync_isocontours<Real>(const_cast<wasm_mesh<Real> &>(m),
                                  const_cast<wasm_ndarray<Real> &>(s),
                                  cut_value);
  });
}

// ============================================================================
// Isocontours — multiple thresholds
// ============================================================================

template <typename Real>
auto extract_cut_values(emscripten::val js_cut_values) -> std::vector<Real> {
  auto n = js_cut_values["length"].as<std::size_t>();
  std::vector<Real> cv(n);
  for (std::size_t i = 0; i < n; ++i)
    cv[i] = js_cut_values[i].as<Real>();
  return cv;
}

template <typename Real>
auto sync_isocontours_multi(wasm_mesh<Real> &mesh, wasm_ndarray<Real> &scalars,
                            emscripten::val js_cut_values)
    -> wasm_curves<Real> {
  auto cv = extract_cut_values<Real>(js_cut_values);
  auto cb = tf::make_isocontours(mesh.polygons_range(), scalars.make_range(),
                                 tf::make_range(cv.data(), cv.size()));
  return wasm_curves<Real>::from_curves_buffer(std::move(cb));
}

template <typename Real>
auto async_isocontours_multi(wasm_mesh<Real> &mesh, wasm_ndarray<Real> &scalars,
                             emscripten::val js_cut_values) -> promise_t {
  auto cv = extract_cut_values<Real>(js_cut_values);
  return promise(
      [m = mesh, s = scalars, cv = std::move(cv)]() -> wasm_curves<Real> {
        auto cb = tf::make_isocontours(
            const_cast<wasm_mesh<Real> &>(m).polygons_range(),
            const_cast<wasm_ndarray<Real> &>(s).make_range(),
            tf::make_range(cv.data(), cv.size()));
        return wasm_curves<Real>::from_curves_buffer(std::move(cb));
      });
}

} // namespace ts
} // namespace tf

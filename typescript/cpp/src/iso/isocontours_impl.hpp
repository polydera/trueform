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

#include "trueform/iso/make_isocontours.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_curves.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <cstddef>
#include <emscripten/val.h>
#include <vector>

namespace tf {
namespace ts {

// ============================================================================
// Main-thread emscripten::val extraction — must run before any async dispatch.
// ============================================================================

template <typename Real>
auto extract_cut_values(emscripten::val js_cut_values) -> std::vector<Real> {
  auto n = js_cut_values["length"].as<std::size_t>();
  std::vector<Real> cv(n);
  for (std::size_t i = 0; i < n; ++i)
    cv[i] = js_cut_values[i].as<Real>();
  return cv;
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

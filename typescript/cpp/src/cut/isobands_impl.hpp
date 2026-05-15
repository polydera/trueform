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

#include "trueform/cut/embedded_isocurves.hpp"
#include "trueform/cut/make_isobands.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_curves.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <cstdint>
#include <emscripten/val.h>
#include <vector>

namespace tf {
namespace ts {

// ============================================================================
// Result types — band mesh / labels / face origins carry the scalar field's
// coordinate dtype on the mesh and curves; labels and face_labels stay int32.
// ============================================================================

template <typename Real> struct isobands_result_t {
  wasm_mesh<Real> mesh;
  wasm_ndarray<int> labels;
  wasm_ndarray<std::int32_t> face_labels;
};

template <typename Real> struct isobands_result_with_curves_t {
  wasm_mesh<Real> mesh;
  wasm_ndarray<int> labels;
  wasm_ndarray<std::int32_t> face_labels;
  wasm_curves<Real> curves;
};

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

inline auto extract_ints(emscripten::val js_arr) -> std::vector<int> {
  auto n = js_arr["length"].as<std::size_t>();
  std::vector<int> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = js_arr[i].as<int>();
  return v;
}

// ============================================================================
// Sync entry points — drive trueform's isocurve / isoband pipeline. Cut values
// and selected-band indices are extracted into POD vectors on the main thread.
// ============================================================================

template <typename Real>
auto sync_isobands(wasm_mesh<Real> &mesh, wasm_ndarray<Real> &scalars,
                   emscripten::val js_cut_values) -> isobands_result_t<Real> {
  auto cv = extract_cut_values<Real>(js_cut_values);
  auto [poly, labels, face_labels] = tf::embedded_isocurves(
      mesh.polygons_range(), scalars.make_range(),
      tf::make_range(cv.data(), cv.size()));
  return {wasm_mesh<Real>::from_polygons_buffer(std::move(poly)),
          wasm_ndarray<int>::from_buffer(std::move(labels)),
          wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels))};
}

template <typename Real>
auto sync_isobands_with_curves(wasm_mesh<Real> &mesh,
                               wasm_ndarray<Real> &scalars,
                               emscripten::val js_cut_values)
    -> isobands_result_with_curves_t<Real> {
  auto cv = extract_cut_values<Real>(js_cut_values);
  auto [poly, labels, face_labels, curves] = tf::embedded_isocurves(
      mesh.polygons_range(), scalars.make_range(),
      tf::make_range(cv.data(), cv.size()), tf::return_curves);
  return {wasm_mesh<Real>::from_polygons_buffer(std::move(poly)),
          wasm_ndarray<int>::from_buffer(std::move(labels)),
          wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels)),
          wasm_curves<Real>::from_curves_buffer(std::move(curves))};
}

template <typename Real>
auto sync_isobands_selected(wasm_mesh<Real> &mesh, wasm_ndarray<Real> &scalars,
                            emscripten::val js_cut_values,
                            emscripten::val js_selected_bands)
    -> isobands_result_t<Real> {
  auto cv = extract_cut_values<Real>(js_cut_values);
  auto sb = extract_ints(js_selected_bands);
  auto [poly, labels, face_labels] = tf::make_isobands(
      mesh.polygons_range(), scalars.make_range(),
      tf::make_range(cv.data(), cv.size()),
      tf::make_range(sb.data(), sb.size()));
  return {wasm_mesh<Real>::from_polygons_buffer(std::move(poly)),
          wasm_ndarray<int>::from_buffer(std::move(labels)),
          wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels))};
}

template <typename Real>
auto sync_isobands_with_curves_selected(wasm_mesh<Real> &mesh,
                                        wasm_ndarray<Real> &scalars,
                                        emscripten::val js_cut_values,
                                        emscripten::val js_selected_bands)
    -> isobands_result_with_curves_t<Real> {
  auto cv = extract_cut_values<Real>(js_cut_values);
  auto sb = extract_ints(js_selected_bands);
  auto [poly, labels, face_labels, curves] = tf::make_isobands(
      mesh.polygons_range(), scalars.make_range(),
      tf::make_range(cv.data(), cv.size()),
      tf::make_range(sb.data(), sb.size()), tf::return_curves);
  return {wasm_mesh<Real>::from_polygons_buffer(std::move(poly)),
          wasm_ndarray<int>::from_buffer(std::move(labels)),
          wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels)),
          wasm_curves<Real>::from_curves_buffer(std::move(curves))};
}

// ============================================================================
// Async — extract emscripten::val on the main thread, then capture by copy
// (shared_ptr refcount++) so worker threads see thread-safe inputs only.
// ============================================================================

template <typename Real>
auto async_isobands(wasm_mesh<Real> &mesh, wasm_ndarray<Real> &scalars,
                    emscripten::val js_cut_values) -> promise_t {
  auto cv = extract_cut_values<Real>(js_cut_values);
  return promise([m = mesh, s = scalars,
                  cv = std::move(cv)]() -> isobands_result_t<Real> {
    auto [poly, labels, face_labels] = tf::embedded_isocurves(
        const_cast<wasm_mesh<Real> &>(m).polygons_range(),
        const_cast<wasm_ndarray<Real> &>(s).make_range(),
        tf::make_range(cv.data(), cv.size()));
    return {wasm_mesh<Real>::from_polygons_buffer(std::move(poly)),
            wasm_ndarray<int>::from_buffer(std::move(labels)),
            wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels))};
  });
}

template <typename Real>
auto async_isobands_with_curves(wasm_mesh<Real> &mesh,
                                wasm_ndarray<Real> &scalars,
                                emscripten::val js_cut_values) -> promise_t {
  auto cv = extract_cut_values<Real>(js_cut_values);
  return promise(
      [m = mesh, s = scalars,
       cv = std::move(cv)]() -> isobands_result_with_curves_t<Real> {
        auto [poly, labels, face_labels, curves] = tf::embedded_isocurves(
            const_cast<wasm_mesh<Real> &>(m).polygons_range(),
            const_cast<wasm_ndarray<Real> &>(s).make_range(),
            tf::make_range(cv.data(), cv.size()), tf::return_curves);
        return {wasm_mesh<Real>::from_polygons_buffer(std::move(poly)),
                wasm_ndarray<int>::from_buffer(std::move(labels)),
                wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels)),
                wasm_curves<Real>::from_curves_buffer(std::move(curves))};
      });
}

template <typename Real>
auto async_isobands_selected(wasm_mesh<Real> &mesh, wasm_ndarray<Real> &scalars,
                             emscripten::val js_cut_values,
                             emscripten::val js_selected_bands) -> promise_t {
  auto cv = extract_cut_values<Real>(js_cut_values);
  auto sb = extract_ints(js_selected_bands);
  return promise([m = mesh, s = scalars, cv = std::move(cv),
                  sb = std::move(sb)]() -> isobands_result_t<Real> {
    auto [poly, labels, face_labels] = tf::make_isobands(
        const_cast<wasm_mesh<Real> &>(m).polygons_range(),
        const_cast<wasm_ndarray<Real> &>(s).make_range(),
        tf::make_range(cv.data(), cv.size()),
        tf::make_range(sb.data(), sb.size()));
    return {wasm_mesh<Real>::from_polygons_buffer(std::move(poly)),
            wasm_ndarray<int>::from_buffer(std::move(labels)),
            wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels))};
  });
}

template <typename Real>
auto async_isobands_with_curves_selected(
    wasm_mesh<Real> &mesh, wasm_ndarray<Real> &scalars,
    emscripten::val js_cut_values,
    emscripten::val js_selected_bands) -> promise_t {
  auto cv = extract_cut_values<Real>(js_cut_values);
  auto sb = extract_ints(js_selected_bands);
  return promise(
      [m = mesh, s = scalars, cv = std::move(cv),
       sb = std::move(sb)]() -> isobands_result_with_curves_t<Real> {
        auto [poly, labels, face_labels, curves] = tf::make_isobands(
            const_cast<wasm_mesh<Real> &>(m).polygons_range(),
            const_cast<wasm_ndarray<Real> &>(s).make_range(),
            tf::make_range(cv.data(), cv.size()),
            tf::make_range(sb.data(), sb.size()), tf::return_curves);
        return {wasm_mesh<Real>::from_polygons_buffer(std::move(poly)),
                wasm_ndarray<int>::from_buffer(std::move(labels)),
                wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels)),
                wasm_curves<Real>::from_curves_buffer(std::move(curves))};
      });
}

} // namespace ts
} // namespace tf

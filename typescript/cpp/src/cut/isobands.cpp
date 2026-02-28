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

#include "trueform/cut/embedded_isocurves.hpp"
#include "trueform/cut/make_isobands.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/cut/result_types.hpp"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <vector>

namespace {

using namespace tf::ts;

auto extract_cut_values(emscripten::val js_cut_values) -> std::vector<float> {
  auto n = js_cut_values["length"].as<std::size_t>();
  std::vector<float> cv(n);
  for (std::size_t i = 0; i < n; ++i)
    cv[i] = js_cut_values[i].as<float>();
  return cv;
}

auto extract_ints(emscripten::val js_arr) -> std::vector<int> {
  auto n = js_arr["length"].as<std::size_t>();
  std::vector<int> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = js_arr[i].as<int>();
  return v;
}

// -- Sync --

auto sync_isobands(wasm_mesh &mesh, wasm_ndarray<float> &scalars,
                   emscripten::val js_cut_values) -> isobands_result {
  auto cv = extract_cut_values(js_cut_values);
  auto [poly, labels] = tf::embedded_isocurves(
      mesh.polygons_range(), scalars.make_range(),
      tf::make_range(cv.data(), cv.size()));
  return {wasm_mesh::from_polygons_buffer(std::move(poly)),
          wasm_ndarray<int>::from_buffer(std::move(labels))};
}

auto sync_isobands_with_curves(wasm_mesh &mesh, wasm_ndarray<float> &scalars,
                               emscripten::val js_cut_values)
    -> isobands_result_with_curves {
  auto cv = extract_cut_values(js_cut_values);
  auto [poly, labels, curves] = tf::embedded_isocurves(
      mesh.polygons_range(), scalars.make_range(),
      tf::make_range(cv.data(), cv.size()), tf::return_curves);
  return {wasm_mesh::from_polygons_buffer(std::move(poly)),
          wasm_ndarray<int>::from_buffer(std::move(labels)),
          wasm_curves::from_curves_buffer(std::move(curves))};
}

// -- Sync (with selected bands) --

auto sync_isobands_selected(wasm_mesh &mesh, wasm_ndarray<float> &scalars,
                            emscripten::val js_cut_values,
                            emscripten::val js_selected_bands)
    -> isobands_result {
  auto cv = extract_cut_values(js_cut_values);
  auto sb = extract_ints(js_selected_bands);
  auto [poly, labels] = tf::make_isobands(
      mesh.polygons_range(), scalars.make_range(),
      tf::make_range(cv.data(), cv.size()),
      tf::make_range(sb.data(), sb.size()));
  return {wasm_mesh::from_polygons_buffer(std::move(poly)),
          wasm_ndarray<int>::from_buffer(std::move(labels))};
}

auto sync_isobands_with_curves_selected(
    wasm_mesh &mesh, wasm_ndarray<float> &scalars,
    emscripten::val js_cut_values, emscripten::val js_selected_bands)
    -> isobands_result_with_curves {
  auto cv = extract_cut_values(js_cut_values);
  auto sb = extract_ints(js_selected_bands);
  auto [poly, labels, curves] = tf::make_isobands(
      mesh.polygons_range(), scalars.make_range(),
      tf::make_range(cv.data(), cv.size()),
      tf::make_range(sb.data(), sb.size()), tf::return_curves);
  return {wasm_mesh::from_polygons_buffer(std::move(poly)),
          wasm_ndarray<int>::from_buffer(std::move(labels)),
          wasm_curves::from_curves_buffer(std::move(curves))};
}

// -- Async --

auto async_isobands(wasm_mesh &mesh, wasm_ndarray<float> &scalars,
                    emscripten::val js_cut_values) -> promise_t {
  auto cv = extract_cut_values(js_cut_values);
  return promise([m = mesh, s = scalars,
                  cv = std::move(cv)]() -> isobands_result {
    auto [poly, labels] = tf::embedded_isocurves(
        const_cast<wasm_mesh &>(m).polygons_range(),
        const_cast<wasm_ndarray<float> &>(s).make_range(),
        tf::make_range(cv.data(), cv.size()));
    return {wasm_mesh::from_polygons_buffer(std::move(poly)),
            wasm_ndarray<int>::from_buffer(std::move(labels))};
  });
}

auto async_isobands_with_curves(wasm_mesh &mesh, wasm_ndarray<float> &scalars,
                                emscripten::val js_cut_values) -> promise_t {
  auto cv = extract_cut_values(js_cut_values);
  return promise(
      [m = mesh, s = scalars,
       cv = std::move(cv)]() -> isobands_result_with_curves {
        auto [poly, labels, curves] = tf::embedded_isocurves(
            const_cast<wasm_mesh &>(m).polygons_range(),
            const_cast<wasm_ndarray<float> &>(s).make_range(),
            tf::make_range(cv.data(), cv.size()), tf::return_curves);
        return {wasm_mesh::from_polygons_buffer(std::move(poly)),
                wasm_ndarray<int>::from_buffer(std::move(labels)),
                wasm_curves::from_curves_buffer(std::move(curves))};
      });
}

auto async_isobands_selected(wasm_mesh &mesh, wasm_ndarray<float> &scalars,
                             emscripten::val js_cut_values,
                             emscripten::val js_selected_bands) -> promise_t {
  auto cv = extract_cut_values(js_cut_values);
  auto sb = extract_ints(js_selected_bands);
  return promise([m = mesh, s = scalars, cv = std::move(cv),
                  sb = std::move(sb)]() -> isobands_result {
    auto [poly, labels] = tf::make_isobands(
        const_cast<wasm_mesh &>(m).polygons_range(),
        const_cast<wasm_ndarray<float> &>(s).make_range(),
        tf::make_range(cv.data(), cv.size()),
        tf::make_range(sb.data(), sb.size()));
    return {wasm_mesh::from_polygons_buffer(std::move(poly)),
            wasm_ndarray<int>::from_buffer(std::move(labels))};
  });
}

auto async_isobands_with_curves_selected(
    wasm_mesh &mesh, wasm_ndarray<float> &scalars,
    emscripten::val js_cut_values,
    emscripten::val js_selected_bands) -> promise_t {
  auto cv = extract_cut_values(js_cut_values);
  auto sb = extract_ints(js_selected_bands);
  return promise(
      [m = mesh, s = scalars, cv = std::move(cv),
       sb = std::move(sb)]() -> isobands_result_with_curves {
        auto [poly, labels, curves] = tf::make_isobands(
            const_cast<wasm_mesh &>(m).polygons_range(),
            const_cast<wasm_ndarray<float> &>(s).make_range(),
            tf::make_range(cv.data(), cv.size()),
            tf::make_range(sb.data(), sb.size()), tf::return_curves);
        return {wasm_mesh::from_polygons_buffer(std::move(poly)),
                wasm_ndarray<int>::from_buffer(std::move(labels)),
                wasm_curves::from_curves_buffer(std::move(curves))};
      });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_isobands) {
  // Result types
  emscripten::value_object<tf::ts::isobands_result>("IsobandsResult")
      .field("mesh", &tf::ts::isobands_result::mesh)
      .field("labels", &tf::ts::isobands_result::labels);

  emscripten::value_object<tf::ts::isobands_result_with_curves>(
      "IsobandsResultWithCurves")
      .field("mesh", &tf::ts::isobands_result_with_curves::mesh)
      .field("labels", &tf::ts::isobands_result_with_curves::labels)
      .field("curves", &tf::ts::isobands_result_with_curves::curves);

  // Sync
  emscripten::function("isobands", &sync_isobands);
  emscripten::function("isobands_with_curves", &sync_isobands_with_curves);
  emscripten::function("isobands_selected", &sync_isobands_selected);
  emscripten::function("isobands_with_curves_selected",
                       &sync_isobands_with_curves_selected);

  // Async
  emscripten::function("dispatch_isobands", &async_isobands);
  emscripten::function("dispatch_isobands_with_curves",
                       &async_isobands_with_curves);
  emscripten::function("dispatch_isobands_selected", &async_isobands_selected);
  emscripten::function("dispatch_isobands_with_curves_selected",
                       &async_isobands_with_curves_selected);
}

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

#include "trueform/core/obb_from.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <emscripten/bind.h>

template <typename Real> struct obb_result_t {
  tf::ts::wasm_ndarray<Real> origin;
  tf::ts::wasm_ndarray<Real> axes;
  tf::ts::wasm_ndarray<Real> extent;
};

namespace {

using namespace tf::ts;

template <typename Real, typename OBB>
auto to_obb_result(const OBB &box) -> obb_result_t<Real> {
  tf::buffer<Real> origin_buf;
  origin_buf.allocate(3);
  for (int i = 0; i < 3; ++i)
    origin_buf[i] = static_cast<Real>(box.origin[i]);

  tf::buffer<Real> axes_buf;
  axes_buf.allocate(9);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      axes_buf[i * 3 + j] = static_cast<Real>(box.axes[i][j]);

  tf::buffer<Real> extent_buf;
  extent_buf.allocate(3);
  for (int i = 0; i < 3; ++i)
    extent_buf[i] = static_cast<Real>(box.extent[i]);

  return {
      wasm_ndarray<Real>::from_buffer(std::move(origin_buf), {3}),
      wasm_ndarray<Real>::from_buffer(std::move(axes_buf), {3, 3}),
      wasm_ndarray<Real>::from_buffer(std::move(extent_buf), {3}),
  };
}

template <typename Real>
auto sync_obb_from(wasm_ndarray<Real> &pts) -> obb_result_t<Real> {
  return to_obb_result<Real>(
      tf::obb_from(tf::make_points<3>(pts.make_range())));
}

template <typename Real>
auto async_obb_from(wasm_ndarray<Real> &pts) -> promise_t {
  return promise([a = pts]() -> obb_result_t<Real> {
    return sync_obb_from<Real>(const_cast<wasm_ndarray<Real> &>(a));
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_obb_float32) {
  emscripten::value_object<obb_result_t<float>>("OBBResultFloat32")
      .field("origin", &obb_result_t<float>::origin)
      .field("axes", &obb_result_t<float>::axes)
      .field("extent", &obb_result_t<float>::extent);

  emscripten::function("obb_from_float32", &sync_obb_from<float>);
  emscripten::function("dispatch_obb_from_float32", &async_obb_from<float>);
}

EMSCRIPTEN_BINDINGS(trueform_obb_float64) {
  emscripten::value_object<obb_result_t<double>>("OBBResultFloat64")
      .field("origin", &obb_result_t<double>::origin)
      .field("axes", &obb_result_t<double>::axes)
      .field("extent", &obb_result_t<double>::extent);

  emscripten::function("obb_from_float64", &sync_obb_from<double>);
  emscripten::function("dispatch_obb_from_float64", &async_obb_from<double>);
}

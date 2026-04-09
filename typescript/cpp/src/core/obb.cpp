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

struct obb_result {
  tf::ts::wasm_ndarray<float> origin;
  tf::ts::wasm_ndarray<float> axes;
  tf::ts::wasm_ndarray<float> extent;
};

namespace {

using namespace tf::ts;

template <typename OBB>
auto to_obb_result(const OBB &box) -> obb_result {
  tf::buffer<float> origin_buf;
  origin_buf.allocate(3);
  for (int i = 0; i < 3; ++i)
    origin_buf[i] = box.origin[i];

  tf::buffer<float> axes_buf;
  axes_buf.allocate(9);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      axes_buf[i * 3 + j] = box.axes[i][j];

  tf::buffer<float> extent_buf;
  extent_buf.allocate(3);
  for (int i = 0; i < 3; ++i)
    extent_buf[i] = box.extent[i];

  return {
      wasm_ndarray<float>::from_buffer(std::move(origin_buf), {3}),
      wasm_ndarray<float>::from_buffer(std::move(axes_buf), {3, 3}),
      wasm_ndarray<float>::from_buffer(std::move(extent_buf), {3}),
  };
}

auto sync_obb_from(wasm_ndarray<float> &pts) -> obb_result {
  return to_obb_result(
      tf::obb_from(tf::make_points<3>(pts.make_range())));
}

auto async_obb_from(wasm_ndarray<float> &pts) -> promise_t {
  return promise([a = pts]() -> obb_result {
    return sync_obb_from(const_cast<wasm_ndarray<float> &>(a));
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_obb) {
  emscripten::value_object<obb_result>("OBBResult")
      .field("origin", &obb_result::origin)
      .field("axes", &obb_result::axes)
      .field("extent", &obb_result::extent);

  emscripten::function("obb_from", &sync_obb_from);
  emscripten::function("dispatch_obb_from", &async_obb_from);
}

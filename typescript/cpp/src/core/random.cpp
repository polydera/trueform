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

#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/random/random.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

auto parse_shape(emscripten::val js_shape) -> tf::small_vector<int, 3> {
  tf::small_vector<int, 3> shape;
  auto len = js_shape["length"].as<int>();
  for (int i = 0; i < len; ++i)
    shape.push_back(js_shape[i].as<int>());
  return shape;
}

template <typename T>
auto random_impl(emscripten::val js_shape, T lo, T hi) -> wasm_ndarray<T> {
  auto shape = parse_shape(js_shape);
  int total = 1;
  for (auto s : shape)
    total *= s;
  tf::buffer<T> buf;
  buf.allocate(total);
  T *dst = buf.data();
  tf::parallel_for_each(
      tf::make_sequence_range(total),
      [=](int i) { dst[i] = tf::random(lo, hi); }, tf::checked);
  return wasm_ndarray<T>::from_buffer(std::move(buf), shape);
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_random) {
  emscripten::function("random_float32", &random_impl<float>);
  emscripten::function("random_float64", &random_impl<double>);
  emscripten::function("random_int32", &random_impl<int>);
}

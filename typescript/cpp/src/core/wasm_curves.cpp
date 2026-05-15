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

#include "trueform/ts/core/wasm_curves.hpp"
#include <emscripten/bind.h>
#include <string>

namespace {

template <typename Real>
auto register_wasm_curves_bindings(const std::string &class_name) -> void {
  using curves_t = tf::ts::wasm_curves<Real>;
  emscripten::class_<curves_t>(class_name.c_str())
      .template constructor<>()
      .class_function("create", &curves_t::create)
      .function("paths", &curves_t::paths)
      .function("points", &curves_t::points)
      .function("size", &curves_t::size)
      .function("destroy", &curves_t::destroy)
      .function("is_valid", &curves_t::is_valid);
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_curves) {
  register_wasm_curves_bindings<float>("NativeFloat32Curves");
  register_wasm_curves_bindings<double>("NativeFloat64Curves");
}

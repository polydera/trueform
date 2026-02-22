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

EMSCRIPTEN_BINDINGS(trueform_curves) {
  emscripten::class_<tf::ts::wasm_curves>("NativeCurves")
      .function("paths", &tf::ts::wasm_curves::paths)
      .function("points", &tf::ts::wasm_curves::points)
      .function("size", &tf::ts::wasm_curves::size)
      .function("destroy", &tf::ts::wasm_curves::destroy)
      .function("is_valid", &tf::ts::wasm_curves::is_valid);
}

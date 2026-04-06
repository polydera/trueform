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

#include "trueform/ts/core/wasm_offset_blocked_buffer.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_offset_blocked_buffer) {
  using obb_int =
      tf::ts::wasm_offset_blocked_buffer<int, int>;

  emscripten::class_<obb_int>("NativeOffsetBlockedIntBuffer")
      .function("offsets", &obb_int::offsets)
      .function("data", &obb_int::data)
      .function("size", &obb_int::size)
      .function("get", &obb_int::get)
      .function("destroy", &obb_int::destroy)
      .function("is_valid", &obb_int::is_valid)
      .class_function("create", &obb_int::create);
}

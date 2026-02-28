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

#include "trueform/ts/core/wasm_index_map.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_index_map) {
  emscripten::value_object<tf::ts::wasm_index_map>("IndexMap")
      .field("f", &tf::ts::wasm_index_map::f)
      .field("keptIds", &tf::ts::wasm_index_map::kept_ids);
}

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

#include "trueform/ts/core/promise.hpp"
#include <emscripten/bind.h>

#include "trueform/ts/core/async_dispatcher.hpp"

static auto retrieve(uintptr_t slot) -> emscripten::val {
  return tf::ts::dispatcher.retrieve(slot);
}

static auto init_tbb() -> tf::ts::promise_t {
  return tf::ts::promise([]() -> int { return 0; });
}

EMSCRIPTEN_BINDINGS(trueform_async) {
  emscripten::function("retrieve", &retrieve);
  emscripten::function("init_tbb", &init_tbb);
}

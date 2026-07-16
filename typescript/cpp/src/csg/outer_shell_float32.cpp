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

#include "./outer_shell_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_csg_outer_shell_float32) {
  using Real = float;
  using namespace tf::ts;

  // Sync
  emscripten::function("outer_shell_float32", &sync_outer_shell<Real>);

  // Async
  emscripten::function("dispatch_outer_shell_float32",
                       &async_outer_shell<Real>);
}

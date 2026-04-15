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
#include "trueform/ts/core/wasm_mesh.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

auto async_ensure(wasm_mesh &m, int what) -> promise_t {
  return promise([a = m, what]() -> int {
    auto &mesh = const_cast<wasm_mesh &>(a);
    switch (what) {
    case 0: mesh.ensure_tree(); break;
    case 1: mesh.normals(); break;
    case 2: mesh.point_normals(); break;
    case 3: mesh.face_membership(); break;
    case 4: mesh.manifold_edge_link(); break;
    case 5: mesh.face_link(); break;
    case 6: mesh.vertex_link(); break;
    default: break;
    }
    return 0;
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_ensure) {
  emscripten::function("dispatch_ensure", &async_ensure);
}

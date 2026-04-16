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
#include "trueform/ts/core/wasm_point_cloud.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

auto async_ensure_pc(wasm_point_cloud &pc) -> promise_t {
  return promise([h = pc]() -> int {
    auto &handle = const_cast<wasm_point_cloud &>(h);
    (void)handle.tree();
    return 0;
  });
}

auto async_ensure(wasm_mesh &m, int what) -> promise_t {
  return promise([h = m, what]() -> int {
    auto &mesh = const_cast<wasm_mesh &>(h);
    switch (what) {
    case 0: (void)mesh.tree();              break;
    case 1: (void)mesh.normals();           break;
    case 2: (void)mesh.point_normals();     break;
    case 3: (void)mesh.face_membership();   break;
    case 4: (void)mesh.manifold_edge_link();break;
    case 5: (void)mesh.face_link();         break;
    case 6: (void)mesh.vertex_link();       break;
    default: break;
    }
    return 0;
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_ensure) {
  emscripten::function("dispatch_ensure", &async_ensure);
  emscripten::function("dispatch_ensure_pc", &async_ensure_pc);
}

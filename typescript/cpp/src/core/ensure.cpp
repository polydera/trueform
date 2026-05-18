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

template <typename Real>
auto async_ensure_pc(wasm_point_cloud<Real> &pc) -> promise_t {
  return promise([h = pc]() -> int {
    auto &handle = const_cast<wasm_point_cloud<Real> &>(h);
    (void)handle.tree();
    return 0;
  });
}

template <typename Real>
auto async_ensure(wasm_mesh<Real> &m, int what) -> promise_t {
  return promise([h = m, what]() -> int {
    auto &mesh = const_cast<wasm_mesh<Real> &>(h);
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
  emscripten::function("dispatch_ensure_float32",    &async_ensure<float>);
  emscripten::function("dispatch_ensure_float64",    &async_ensure<double>);
  emscripten::function("dispatch_ensure_pc_float32", &async_ensure_pc<float>);
  emscripten::function("dispatch_ensure_pc_float64", &async_ensure_pc<double>);
}

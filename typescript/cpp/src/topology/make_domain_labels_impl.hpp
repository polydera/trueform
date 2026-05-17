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
#pragma once
#include <emscripten/bind.h>

#include "trueform/topology/domain_config.hpp"
#include "trueform/topology/make_domain_labels.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"

namespace tf {
namespace ts {

// Templated on Real only so that each per-real binding registers a
// distinct value_object type under a distinct JS name. Fields are not
// Real-dependent.
template <typename Real> struct domain_labels_result_t {
  wasm_ndarray<int> labels; // shape [n_faces, 2]
  int n_domains;
  int outer_shell_label;
};

template <typename Real>
auto sync_make_domain_labels(wasm_mesh<Real> &m, int config)
    -> domain_labels_result_t<Real> {
  auto poly = m.polygons_range();
  auto dl = tf::make_domain_labels(poly,
                                   static_cast<tf::domain_config>(config));

  auto flat = std::move(dl.labels.data_buffer());
  auto n_faces = static_cast<int>(flat.size() / 2);

  domain_labels_result_t<Real> result;
  result.labels =
      wasm_ndarray<int>::from_buffer(std::move(flat), {n_faces, 2});
  result.n_domains = static_cast<int>(dl.n_domains);
  result.outer_shell_label = static_cast<int>(dl.outer_shell_label);
  return result;
}

template <typename Real>
auto async_make_domain_labels(wasm_mesh<Real> &m, int config) -> promise_t {
  return promise([a = m, c = config]() -> domain_labels_result_t<Real> {
    return sync_make_domain_labels<Real>(const_cast<wasm_mesh<Real> &>(a), c);
  });
}

} // namespace ts
} // namespace tf

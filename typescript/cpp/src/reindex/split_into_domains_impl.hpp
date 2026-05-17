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
#include <algorithm>
#include <emscripten/bind.h>
#include <vector>

#include "trueform/reindex.hpp"
#include "trueform/topology/domain_labels.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"

namespace tf {
namespace ts {

// ============================================================================
// Result type — one watertight, outward-oriented submesh per domain plus
// the per-component domain ids.
// ============================================================================

template <typename Real> struct split_domains_result_t {
  std::vector<wasm_mesh<Real>> components;
  wasm_ndarray<int> labels;
};

// ============================================================================
// split_into_domains(mesh, labels, n_domains) — per-side polygon split keyed
// by tf::domain_labels.  outer_shell_label is intentionally not threaded
// through: it is not consumed by tf::split_into_domains.
// ============================================================================

template <typename Real>
auto compute_split_into_domains(wasm_mesh<Real> &m, wasm_ndarray<int> &labels,
                                int n_domains)
    -> split_domains_result_t<Real> {
  auto poly = m.polygons_range();

  tf::domain_labels<int> dl;
  dl.n_domains = n_domains;
  auto labels_range = labels.make_range();
  dl.labels.allocate(labels_range.size() / 2);
  std::copy(labels_range.begin(), labels_range.end(),
            dl.labels.data_buffer().begin());

  auto [components, comp_labels] = tf::split_into_domains(poly, dl);

  split_domains_result_t<Real> result;
  result.components.reserve(components.size());
  for (auto &c : components)
    result.components.push_back(
        wasm_mesh<Real>::from_polygons_buffer(std::move(c)));
  auto n_labels = static_cast<int>(comp_labels.size());
  result.labels =
      wasm_ndarray<int>::from_buffer(std::move(comp_labels), {n_labels});
  return result;
}

template <typename Real>
auto sync_split_into_domains(wasm_mesh<Real> &m, wasm_ndarray<int> &labels,
                             int n_domains) -> split_domains_result_t<Real> {
  return compute_split_into_domains<Real>(m, labels, n_domains);
}

template <typename Real>
auto async_split_into_domains(wasm_mesh<Real> &m, wasm_ndarray<int> &labels,
                              int n_domains) -> promise_t {
  return promise([a = m, b = labels, n = n_domains]()
                     -> split_domains_result_t<Real> {
    return compute_split_into_domains<Real>(
        const_cast<wasm_mesh<Real> &>(a),
        const_cast<wasm_ndarray<int> &>(b), n);
  });
}

} // namespace ts
} // namespace tf

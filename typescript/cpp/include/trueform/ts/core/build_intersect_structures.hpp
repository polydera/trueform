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

#include "trueform/ts/core/wasm_mesh.hpp"
#include <tbb/parallel_invoke.h>
#include <tbb/task_group.h>

namespace tf {
namespace ts {

template <typename Real>
auto needs_intersect_structures(const wasm_mesh<Real> &m) -> bool {
  return !m.is_tree_fresh() || !m.is_face_membership_fresh() ||
         !m.is_manifold_edge_link_fresh();
}

template <typename Real>
auto build_intersect_structures(wasm_mesh<Real> &m) -> void {
  bool need_tree = !m.is_tree_fresh();
  bool need_topo =
      !m.is_face_membership_fresh() || !m.is_manifold_edge_link_fresh();
  auto build_topo = [&] {
    m.build_face_membership();
    m.build_manifold_edge_link();
  };
  if (need_tree && need_topo)
    tbb::parallel_invoke([&] { m.build_tree(); }, build_topo);
  else if (need_tree)
    m.build_tree();
  else if (need_topo)
    build_topo();
}

template <typename Real>
auto build_intersect_structures(wasm_mesh<Real> &a, wasm_mesh<Real> &b)
    -> void {
  if (a.same_data(b)) {
    build_intersect_structures(a);
    return;
  }
  bool na = needs_intersect_structures(a);
  bool nb = needs_intersect_structures(b);
  if (na && nb)
    tbb::parallel_invoke([&] { build_intersect_structures(a); },
                         [&] { build_intersect_structures(b); });
  else if (na)
    build_intersect_structures(a);
  else if (nb)
    build_intersect_structures(b);
}

// Precondition: meshes are distinct (no two entries share one mesh_data). We do
// not dedup (a pairwise same_data check would be O(n^2)).
template <typename Meshes>
auto build_intersect_structures_all(Meshes &meshes) -> void {
  tbb::task_group tg;
  bool any = false;
  for (auto &m : meshes)
    if (needs_intersect_structures(m)) {
      tg.run([&m] { build_intersect_structures(m); });
      any = true;
    }
  if (any)
    tg.wait();
}

} // namespace ts
} // namespace tf

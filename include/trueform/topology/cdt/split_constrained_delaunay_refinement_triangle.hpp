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
#include "./queue_constrained_delaunay_refinement_lawson_edge.hpp"
#include "./queue_constrained_delaunay_refinement_triangle.hpp"
#include "./restore_constrained_delaunay_refinement.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt {

template <typename Owner>
auto split_constrained_delaunay_refinement_triangle(
    Owner &owner, typename Owner::index_type face,
    typename Owner::index_type point) -> void {
  using Index = typename Owner::index_type;

  Index v0 = owner._t[face].v[0], v1 = owner._t[face].v[1],
        v2 = owner._t[face].v[2];
  Index n0 = owner._t[face].n[0], n1 = owner._t[face].n[1],
        n2 = owner._t[face].n[2];
  Index s0 = owner._t[face].seg[0], s1 = owner._t[face].seg[1],
        s2 = owner._t[face].seg[2];
  Index label = owner._label[std::size_t(face)];
  Index ta = Index(owner._t.size());
  Index tb = ta + 1;
  std::uint32_t stamp = owner._t[face].stamp + 1;
  owner._t[face] = {{v0, v1, point},
                    {n0, ta, tb},
                    {s0, Owner::none, Owner::none},
                    stamp};
  owner._t.push_back({{v1, v2, point},
                      {n1, tb, face},
                      {s1, Owner::none, Owner::none},
                      0});
  owner._t.push_back({{v2, v0, point},
                      {n2, face, ta},
                      {s2, Owner::none, Owner::none},
                      0});
  owner._label.push_back(label);
  owner._label.push_back(label);
  if (n1 != Owner::none) {
    int k = owner.edge_back(n1, face);
    if (k >= 0)
      owner._t[n1].n[k] = ta;
  }
  if (n2 != Owner::none) {
    int k = owner.edge_back(n2, face);
    if (k >= 0)
      owner._t[n2].n[k] = tb;
  }
  queue_constrained_delaunay_refinement_lawson_edge(owner, face, 0);
  queue_constrained_delaunay_refinement_lawson_edge(owner, ta, 0);
  queue_constrained_delaunay_refinement_lawson_edge(owner, tb, 0);
  queue_constrained_delaunay_refinement_triangle(owner, face);
  queue_constrained_delaunay_refinement_triangle(owner, ta);
  queue_constrained_delaunay_refinement_triangle(owner, tb);
  restore_constrained_delaunay_refinement(owner, point);
}

} // namespace tf::topology::cdt

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
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto restore_constrained_delaunay_refinement(
    Owner &owner, typename Owner::index_type fresh) -> void {
  using Index = typename Owner::index_type;

  while (owner._lawson.size() != 0) {
    Index packed = owner._lawson.back();
    owner._lawson.pop_back();
    Index f = packed >> 2;
    int e = packed & 3;
    Index g = owner._t[f].n[e];
    Index a = owner._t[f].v[e];
    Index b = owner._t[f].v[(e + 1) % 3];
    Index c = owner._t[f].v[(e + 2) % 3];
    if (owner.constrained(f, e)) {
      if (fresh != a && fresh != b && owner.splittable(f, e) &&
          !owner.constraint_connected(fresh, a) &&
          !owner.constraint_connected(fresh, b)) {
        if (owner.encroaches(owner._dp[std::size_t(a)],
                             owner._dp[std::size_t(b)],
                             owner._dp[std::size_t(fresh)]))
          owner._pending.push_back({f, Index(e), owner._t[f].stamp});
      }
      continue;
    }
    if (g == Owner::none)
      continue;
    int eb = owner.edge_back(g, f);
    Index d = owner._t[g].v[(eb + 2) % 3];
    if (c == d)
      continue;
    if (owner.incircle(a, b, c, d) <= 0)
      continue;
    if (owner.orient(a, d, c) <= 0 || owner.orient(b, c, d) <= 0)
      continue;

    Index nb_bc = owner._t[f].n[(e + 1) % 3];
    Index nb_ca = owner._t[f].n[(e + 2) % 3];
    Index nb_ad = owner._t[g].n[(eb + 1) % 3];
    Index nb_db = owner._t[g].n[(eb + 2) % 3];
    Index s_bc = owner._t[f].seg[(e + 1) % 3];
    Index s_ca = owner._t[f].seg[(e + 2) % 3];
    Index s_ad = owner._t[g].seg[(eb + 1) % 3];
    Index s_db = owner._t[g].seg[(eb + 2) % 3];

    owner._t[f].v[0] = a;
    owner._t[f].v[1] = d;
    owner._t[f].v[2] = c;
    owner._t[f].n[0] = nb_ad;
    owner._t[f].n[1] = g;
    owner._t[f].n[2] = nb_ca;
    owner._t[f].seg[0] = s_ad;
    owner._t[f].seg[1] = Owner::none;
    owner._t[f].seg[2] = s_ca;
    owner._t[g].v[0] = b;
    owner._t[g].v[1] = c;
    owner._t[g].v[2] = d;
    owner._t[g].n[0] = nb_bc;
    owner._t[g].n[1] = f;
    owner._t[g].n[2] = nb_db;
    owner._t[g].seg[0] = s_bc;
    owner._t[g].seg[1] = Owner::none;
    owner._t[g].seg[2] = s_db;
    if (nb_ad != Owner::none) {
      int k = owner.edge_back(nb_ad, g);
      if (k >= 0)
        owner._t[nb_ad].n[k] = f;
    }
    if (nb_bc != Owner::none) {
      int k = owner.edge_back(nb_bc, f);
      if (k >= 0)
        owner._t[nb_bc].n[k] = g;
    }
    ++owner._t[f].stamp;
    ++owner._t[g].stamp;
    queue_constrained_delaunay_refinement_lawson_edge(owner, f, 0);
    queue_constrained_delaunay_refinement_lawson_edge(owner, f, 2);
    queue_constrained_delaunay_refinement_lawson_edge(owner, g, 0);
    queue_constrained_delaunay_refinement_lawson_edge(owner, g, 2);
    queue_constrained_delaunay_refinement_triangle(owner, f);
    queue_constrained_delaunay_refinement_triangle(owner, g);
  }
}

} // namespace tf::topology::cdt

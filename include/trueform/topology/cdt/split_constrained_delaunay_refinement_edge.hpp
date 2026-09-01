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
#include <utility>

namespace tf::topology::cdt {

template <typename Owner>
auto split_constrained_delaunay_refinement_edge(
    Owner &owner, typename Owner::index_type face, int edge,
    typename Owner::index_type point,
    typename Owner::index_type segment_lo,
    typename Owner::index_type segment_hi) -> void {
  using Index = typename Owner::index_type;

  Index g = owner._t[face].n[edge];
  Index a = owner._t[face].v[edge];
  Index b = owner._t[face].v[(edge + 1) % 3];
  Index c = owner._t[face].v[(edge + 2) % 3];
  Index nb_bc = owner._t[face].n[(edge + 1) % 3];
  Index nb_ca = owner._t[face].n[(edge + 2) % 3];
  Index s_bc = owner._t[face].seg[(edge + 1) % 3];
  Index s_ca = owner._t[face].seg[(edge + 2) % 3];
  Index lab_f = owner._label[std::size_t(face)];
  Index s_ap = segment_lo, s_pb = segment_hi;
  if (segment_lo != Owner::none &&
      owner._segments[std::size_t(segment_lo)].lo_vertex != a)
    std::swap(s_ap, s_pb);

  Index fa = Index(owner._t.size());
  std::uint32_t stamp = owner._t[face].stamp + 1;
  owner._t[face] = {{a, point, c},
                    {Owner::none, fa, nb_ca},
                    {s_ap, Owner::none, s_ca},
                    stamp};
  owner._t.push_back({{point, b, c},
                      {Owner::none, nb_bc, face},
                      {s_pb, s_bc, Owner::none},
                      0});
  owner._label.push_back(lab_f);
  if (nb_bc != Owner::none) {
    int k = owner.edge_back(nb_bc, face);
    if (k >= 0)
      owner._t[nb_bc].n[k] = fa;
  }
  owner._t[face].n[1] = fa;
  owner._t[fa].n[2] = face;

  if (g != Owner::none) {
    int eb = owner.edge_back(g, face);
    Index d = owner._t[g].v[(eb + 2) % 3];
    Index nb_ad = owner._t[g].n[(eb + 1) % 3];
    Index nb_db = owner._t[g].n[(eb + 2) % 3];
    Index s_ad = owner._t[g].seg[(eb + 1) % 3];
    Index s_db = owner._t[g].seg[(eb + 2) % 3];
    Index lab_g = owner._label[std::size_t(g)];
    Index ga = Index(owner._t.size());
    std::uint32_t gstamp = owner._t[g].stamp + 1;
    owner._t[g] = {{b, point, d},
                   {Owner::none, ga, nb_db},
                   {s_pb, Owner::none, s_db},
                   gstamp};
    owner._t.push_back({{point, a, d},
                        {Owner::none, nb_ad, g},
                        {s_ap, s_ad, Owner::none},
                        0});
    owner._label.push_back(lab_g);
    if (nb_ad != Owner::none) {
      int k = owner.edge_back(nb_ad, g);
      if (k >= 0)
        owner._t[nb_ad].n[k] = ga;
    }
    owner._t[face].n[0] = ga;
    owner._t[ga].n[0] = face;
    owner._t[fa].n[0] = g;
    owner._t[g].n[0] = fa;
    queue_constrained_delaunay_refinement_lawson_edge(owner, g, 2);
    queue_constrained_delaunay_refinement_lawson_edge(owner, ga, 1);
    queue_constrained_delaunay_refinement_triangle(owner, g);
    queue_constrained_delaunay_refinement_triangle(owner, ga);
  }
  queue_constrained_delaunay_refinement_lawson_edge(owner, face, 2);
  queue_constrained_delaunay_refinement_lawson_edge(owner, fa, 1);
  queue_constrained_delaunay_refinement_triangle(owner, face);
  queue_constrained_delaunay_refinement_triangle(owner, fa);
  restore_constrained_delaunay_refinement(owner, point);
}

} // namespace tf::topology::cdt

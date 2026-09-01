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
#include "./find_constrained_delaunay_refinement_constraint.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto adopt_constrained_delaunay_refinement_faces(Owner &owner) -> void {
  using Index = typename Owner::index_type;

  const bool broadcast = owner._constraint_alias_blocks.size() != 0;
  const auto n_faces = std::size_t(owner._cdt.n_triangles());
  owner._t.reserve(n_faces);
  owner._label.reserve(n_faces);
  // The CDT's DCEL already carries the adjacency: neighbors across opp,
  // constraint flags on edges. Its face handedness is uniform
  // (opp-consistent structure), so one orientation probe decides the vertex
  // order for every face.
  int handed = 0;
  owner._cdt.for_each_face_adjacency(
      [&](Index i, Index a, Index b, Index c, Index label,
          const std::array<Index, 3> &nbrs,
          const std::array<bool, 3> &cons) {
        if (handed == 0)
          handed = owner.orient(a, b, c) < 0 ? -1 : 1;
        typename Owner::triangle tt{};
        tt.stamp = 0;
        if (handed > 0) {
          tt.v[0] = a;
          tt.v[1] = b;
          tt.v[2] = c;
          for (int e = 0; e < 3; ++e) {
            tt.n[e] = nbrs[std::size_t(e)];
            tt.seg[e] = Owner::none;
          }
          for (int e = 0; e < 3; ++e)
            if (cons[std::size_t(e)])
              tt.seg[e] = find_constrained_delaunay_refinement_constraint(
                  owner, tt.v[e], tt.v[(e + 1) % 3], broadcast);
        } else {
          // Reversed: (a, c, b); edge k of the new order is the reverse of
          // DCEL edge (2 - k).
          tt.v[0] = a;
          tt.v[1] = c;
          tt.v[2] = b;
          for (int e = 0; e < 3; ++e) {
            tt.n[e] = nbrs[std::size_t(2 - e)];
            tt.seg[e] = Owner::none;
          }
          for (int e = 0; e < 3; ++e)
            if (cons[std::size_t(2 - e)])
              tt.seg[e] = find_constrained_delaunay_refinement_constraint(
                  owner, tt.v[e], tt.v[(e + 1) % 3], broadcast);
        }
        (void)i;
        owner._t.push_back(tt);
        owner._label.push_back(label);
      });
}

} // namespace tf::topology::cdt

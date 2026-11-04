/*
 * Copyright (c) 2026 XLAB
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
#include "../../../exact/meta.hpp"
#include "../../../exact/orient3d.hpp"
#include "./recovery/tet_pierce_in_facet.hpp"
#include "./tet_deepest_of.hpp"
#include "./tet_flip_constraints.hpp"
#include "./tet_triangle_edge_contact.hpp"
#include "./tet_with_sites.hpp"
#include <cstddef>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// Whether a flip may be made under `fc`: `kind` 1 is the 2-3 of the facet
/// a b c between the apexes d and e, making the edge e d and the facets
/// e d a, e d b, e d c; `kind` 2 is the 3-2 of the edge e d about the ring
/// facets e d a, e d b, e d c, making the facet a b c. A 2-3 is refused when
/// a new facet or the new edge crosses the segment, the new edge pierces the
/// polygon inside its rim, or the new edge holds the removed vertex; a 3-2 when
/// the new facet's interior or, coplanar, its boundary crosses the segment. `c`
/// may be the infinite corner; no other may. Each contact is read at the
/// deepest depth among its five sites, the pierce at the deepest of d and e,
/// the polygon's corners being build sites on the frame.
template <typename Owner>
auto tet_flip_eligible(const Owner &owner, int kind,
                       typename Owner::index_type a,
                       typename Owner::index_type b,
                       typename Owner::index_type c,
                       typename Owner::index_type d,
                       typename Owner::index_type e,
                       const tet_flip_constraints<typename Owner::index_type,
                                                  typename Owner::coord_type,
                                                  typename Owner::int_type> &fc)
    -> bool {
  using Index = typename Owner::index_type;
  tet_triangle_edge_contact contact{};
  const auto classify = [&owner, &fc, &contact](Index x, Index y, Index z) {
    const Index s0 = fc.segment[0], s1 = fc.segment[1];
    return tet_with_sites(
        owner, tet_deepest_of(owner, x, y, z, s0, s1), [&](const auto &sites) {
          using lattice = typename std::decay_t<decltype(sites)>::int_type;
          return classify_tet_triangle_edge<lattice>(
              sites[std::size_t(x)].pt, sites[std::size_t(y)].pt,
              sites[std::size_t(z)].pt, sites[std::size_t(s0)].pt,
              sites[std::size_t(s1)].pt, contact);
        });
  };
  if (fc.segment[0] != Index(-1)) {
    if (kind == 1) {
      const Index corners[3] = {a, b, c};
      for (std::size_t i = 0; i < 3; ++i) {
        if (corners[i] == Owner::infinite)
          continue;
        const int met = classify(e, d, corners[i]);
        if (met == 2) {
          if (contact.type[0] == tet_contact::across_face)
            return false;
          if (contact.type[0] == tet_contact::across_edge &&
              contact.pos[0] == 0)
            return false;
        } else if (met == 4) {
          if (contact.type[0] == tet_contact::across_edge &&
              contact.pos[0] == 0)
            return false;
        }
      }
    } else if (c != Owner::infinite) {
      const int met = classify(a, b, c);
      if (met == 2) {
        if (contact.type[0] == tet_contact::across_face)
          return false;
      } else if (met == 4) {
        if (contact.type[0] == tet_contact::across_edge)
          return false;
      }
    }
  }
  if (fc.polygon.size() && kind == 1 &&
      tet_with_sites(
          owner, tet_deepest_of(owner, d, e), [&](const auto &sites) {
            using lattice = typename std::decay_t<decltype(sites)>::int_type;
            using T2 = typename tf::exact::meta<lattice>::T2;
            const auto &first = fc.polygon[0];
            const T2 oe = tf::exact::orient3d_value_scaled<lattice>(
                sites[std::size_t(first[0])].pt,
                sites[std::size_t(first[1])].pt,
                sites[std::size_t(first[2])].pt, sites[std::size_t(e)].pt);
            const T2 od = tf::exact::orient3d_value_scaled<lattice>(
                sites[std::size_t(first[0])].pt,
                sites[std::size_t(first[1])].pt,
                sites[std::size_t(first[2])].pt, sites[std::size_t(d)].pt);
            return ((oe > T2(0) && od < T2(0)) || (oe < T2(0) && od > T2(0))) &&
                   tet_pierce_in_facet(sites, fc.polygon, fc.axes, e, d, oe,
                                       od);
          }))
    return false;
  if (fc.removed != Index(-1) && kind == 1)
    if (d == fc.removed || e == fc.removed)
      return false;
  return true;
}

} // namespace tf::topology::cdt::dt3

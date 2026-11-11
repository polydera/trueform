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
#include "../../../../exact/meta.hpp"
#include "../tet_deepest_of.hpp"
#include "../tet_with_sites.hpp"
#include "../tetrahedralization_owner.hpp"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// The numerator over 2^bits at which the segment from site `a` to site `b`
/// meets the line through the sites `c` and `d`, the four coplanar, read at
/// the deepest depth among them: the meeting is where a b crosses a plane
/// holding c d, the one through c d and an axis that a b crosses steepest.
/// `-1` where the lines are parallel, a b meets the plane behind `a`, or
/// meets it between two numerators of the frame: the meeting a frame cannot
/// place exactly is no split of it.
template <typename Index, typename Coord, typename Int>
auto tet_segment_crossing_numerator(
    const tetrahedralization_owner<Index, Coord, Int> &owner, Index a, Index b,
    Index c, Index d, int bits) -> std::int64_t {
  return tet_with_sites(
      owner, tet_deepest_of(owner, a, b, c, d), [&](const auto &sites) {
        using lattice = typename std::decay_t<decltype(sites)>::int_type;
        using T2 = typename tf::exact::meta<lattice>::T2;
        using T3 = typename tf::exact::meta<lattice>::T3;
        const auto pa = sites[std::size_t(a)].pt;
        const auto pb = sites[std::size_t(b)].pt;
        const auto pc = sites[std::size_t(c)].pt;
        const auto pd = sites[std::size_t(d)].pt;
        T2 v[3], e[3];
        for (std::size_t k = 0; k < 3; ++k) {
          v[k] = T2(pd[k]) - T2(pc[k]);
          e[k] = T2(pb[k]) - T2(pa[k]);
        }
        // the plane's normal is v x axis; a b crosses it steepest where the
        // normal's reach along a b is largest
        std::size_t axis = 0;
        T2 steepest = 0;
        T2 n[3] = {0, 0, 0};
        for (std::size_t k = 0; k < 3; ++k) {
          const std::size_t u = (k + 1) % 3, w = (k + 2) % 3;
          const T2 reach = e[u] * v[w] - e[w] * v[u];
          const T2 size = reach < 0 ? -reach : reach;
          if (size > steepest) {
            steepest = size;
            axis = k;
          }
        }
        if (steepest == T2(0))
          return std::int64_t(-1);
        {
          const std::size_t u = (axis + 1) % 3, w = (axis + 2) % 3;
          n[u] = v[w];
          n[w] = -v[u];
        }
        T2 va = 0, vb = 0;
        for (std::size_t k = 0; k < 3; ++k) {
          va += (T2(pa[k]) - T2(pc[k])) * n[k];
          vb += (T2(pb[k]) - T2(pc[k])) * n[k];
        }
        T3 num = T3(va);
        T3 den = T3(va) - T3(vb);
        if (den == T3(0))
          return std::int64_t(-1);
        if (den < T3(0)) {
          num = -num;
          den = -den;
        }
        if (num < T3(0))
          return std::int64_t(-1);
        num = num * (T3(1) << bits);
        if (num % den != T3(0))
          return std::int64_t(-1);
        return std::int64_t(num / den);
      });
}

} // namespace tf::topology::cdt::dt3

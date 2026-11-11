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
#include "../../../../../core/algorithm/parallel_for_each.hpp"
#include "../../../../../core/buffer.hpp"
#include "../../../../../core/checked.hpp"
#include "../../../../../core/small_vector.hpp"
#include "../../../../../core/views/sequence_range.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Per facet of `facets`, a range of triangle ids each, the patch most of
/// its triangles hold in `patch_of_triangle`, the lower patch on a tie,
/// into `patch_of_facet`; `-1` for a facet of no triangle.
template <typename Facets, typename Index>
auto make_tet_facet_patches(const Facets &facets,
                            const tf::buffer<Index> &patch_of_triangle,
                            tf::buffer<Index> &patch_of_facet) -> void {
  using tally_t = tf::small_vector<std::array<Index, 2>, 8>;
  patch_of_facet.allocate(facets.size());
  tf::parallel_for_each(
      tf::make_sequence_range(facets.size()),
      [&facets, &patch_of_triangle, &patch_of_facet](std::size_t f,
                                                     tally_t &tally) {
        tally.clear();
        for (auto t : facets[f]) {
          const Index p = patch_of_triangle[std::size_t(t)];
          auto at = tally.begin();
          while (at != tally.end() && (*at)[0] != p)
            ++at;
          if (at == tally.end())
            tally.push_back({p, Index(1)});
          else
            ++(*at)[1];
        }
        Index best = Index(-1), most = 0;
        for (const auto &[p, n] : tally)
          if (n > most || (n == most && p < best)) {
            best = p;
            most = n;
          }
        patch_of_facet[f] = best;
      },
      tally_t{}, tf::checked);
}

} // namespace tf::topology::cdt::dt3

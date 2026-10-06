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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/delaunay_tetrahedralizer.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::fill {

/// Every facet of the complex once, as an ascending triple of rim positions,
/// the triples in ascending order.
///
/// Both cells of a facet name it, so one of them writes it: the lower of the
/// two, and the finite one where the hull closes the facet. The adjacency the
/// complex already published is what states that, so no key is deduplicated —
/// a cell counts what it owns, the counts prefix into `owned`, and every
/// facet is written at a ticket of its own.
template <typename Index, typename Coord, typename Int, typename ExecutionPolicy>
auto gather_hole_facets(
    const tf::delaunay_tetrahedralizer<Index, Coord, Int, ExecutionPolicy> &dt,
    const tf::buffer<Index> &position_of_site, tf::buffer<Index> &owned,
    tf::buffer<std::array<Index, 3>> &facets) -> void {
  const auto neighbors = dt.neighbors();
  const auto tets = dt.tets();
  const Index none = dt.k_none;
  const std::size_t cells = dt.n_tets();

  owned.allocate(cells + 1);
  owned[0] = Index(0);
  tf::parallel_for_each(
      tf::make_sequence_range(cells),
      [&neighbors, &owned, none](std::size_t cell) {
        Index count = 0;
        for (std::size_t drop = 0; drop < 4; ++drop) {
          const Index peer = neighbors[cell][drop];
          if (peer == none || peer > Index(cell))
            ++count;
        }
        owned[cell + 1] = count;
      },
      tf::checked);
  for (std::size_t cell = 0; cell < cells; ++cell)
    owned[cell + 1] += owned[cell];

  facets.allocate(std::size_t(owned[cells]));
  tf::parallel_for_each(
      tf::make_sequence_range(cells),
      [&neighbors, &tets, &owned, &position_of_site, &facets,
       none](std::size_t cell) {
        const auto corners = tets[cell];
        std::size_t at = std::size_t(owned[cell]);
        for (std::size_t drop = 0; drop < 4; ++drop) {
          const Index peer = neighbors[cell][drop];
          if (peer != none && peer < Index(cell))
            continue;
          std::array<Index, 3> facet{};
          std::size_t corner = 0;
          for (std::size_t slot = 0; slot < 4; ++slot)
            if (slot != drop)
              facet[corner++] = position_of_site[std::size_t(corners[slot])];
          std::sort(facet.begin(), facet.end());
          facets[at++] = facet;
        }
      },
      tf::checked);
  tbb::parallel_sort(facets.begin(), facets.end());
}

} // namespace tf::fill

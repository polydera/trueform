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
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/points.hpp"
#include "../../../core/range.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "../../../exact/vertex.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Load positions and names with their input slots for welding.
template <bool Parallel = false, typename Owner, typename PointsPolicy>
auto load_tetrahedralization_sites(Owner &owner,
                                   const tf::points<PointsPolicy> &points)
    -> void {
  using Index = typename Owner::index_type;
  using Coord = typename Owner::coord_type;
  owner._site_scratch.allocate(points.size());
  const auto load_site = [&owner, &points](std::size_t i) {
    const auto &p = points[i];
    owner._site_scratch[i] = {
        tf::exact::pt3<Coord>{Coord(p[0]), Coord(p[1]), Coord(p[2])}, Index(i),
        Index(i)};
  };
  if constexpr (Parallel)
    tf::parallel_for_each(tf::make_sequence_range(points.size()), load_site,
                          tf::checked);
  else
    for (std::size_t i = 0; i < points.size(); ++i)
      load_site(i);
}

template <bool Parallel = false, typename Owner, typename Iterator>
auto load_tetrahedralization_sites(
    Owner &owner, const tf::range<Iterator, tf::dynamic_size> &sites) -> void {
  using Index = typename Owner::index_type;
  owner._site_scratch.allocate(sites.size());
  const auto load_site = [&owner, &sites](std::size_t i) {
    owner._site_scratch[i] = {sites[i].pt, sites[i].id, Index(i)};
  };
  if constexpr (Parallel)
    tf::parallel_for_each(tf::make_sequence_range(sites.size()), load_site,
                          tf::checked);
  else
    for (std::size_t i = 0; i < sites.size(); ++i)
      load_site(i);
}

} // namespace tf::topology::cdt::dt3

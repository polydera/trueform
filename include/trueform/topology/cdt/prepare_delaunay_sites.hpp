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
#include "./load_delaunay_sites.hpp"
#include "./load_delaunay_sites_parallel.hpp"
#include "./sort_delaunay_sites_morton.hpp"
#include "./weld_delaunay_sites.hpp"
#include <cstddef>

namespace tf::topology::cdt {

/// Convert, physically Morton-sort, and optionally exact-weld the shared site
/// carrier. `BuildIndexMap` records duplicate representatives; final output
/// indices are assigned only after the caller freezes its output vertex order.
/// `Weld=false` is for an already-welded retained point carrier.
template <bool Weld = true, bool ParallelPreparation = false,
          bool BuildIndexMap = false, typename Owner, typename Points,
          typename Converter>
auto prepare_delaunay_sites(Owner &owner, const Points &points,
                            const Converter &converter) -> void {
  owner._sites.clear();
  owner._site_scratch.clear();
  owner._keys.clear();
  owner._key_scratch.clear();
  owner._orientation_fits_t1 = false;
  owner._incircle_inner_fits_t1 = false;
  if constexpr (BuildIndexMap) {
    owner._index_map.f().allocate(points.size());
    for (std::size_t i = 0; i < points.size(); ++i)
      owner._index_map.f()[i] = static_cast<typename Owner::index_type>(i);
  }
  if (points.size() == 0)
    return;
  if constexpr (ParallelPreparation) {
    load_delaunay_sites_parallel(owner, points, converter);
    sort_delaunay_sites_morton_parallel(owner);
  } else {
    load_delaunay_sites(owner, points, converter);
    sort_delaunay_sites_morton(owner);
  }
  if constexpr (Weld)
    weld_delaunay_sites<BuildIndexMap>(owner);
}

} // namespace tf::topology::cdt

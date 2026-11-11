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
#include "../../../../../core/views/mapped_range.hpp"
#include "../../../../../core/views/offset_block_range.hpp"
#include "../../../../../core/views/sequence_range.hpp"
#include "../../../../make_applier.hpp"
#include "../../tet_execution_tuning.hpp"
#include "../../tet_site_position.hpp"
#include "./make_tet_facet_patches.hpp"
#include "./make_tet_first_facet_patches.hpp"
#include "./partition_tet_surface.hpp"
#include "./tet_patch_workspace.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The regions of one opening: per triangle, facet and site the region
/// it belongs to, `-1` where none does (@ref partition_tet_surface,
/// `regions` is 1 on a surface below twice the target,
/// where every table is identically 0 and a round states no boundary at all.
/// `facet_first` prefixes the facets over the triangles, which is the id
/// space the link the phase is given is in.
template <typename Index> struct tet_patch_tables {
  tf::buffer<Index> triangle;
  tf::buffer<Index> facet;
  tf::buffer<Index> site;
  tf::buffer<Index> facet_first;
  Index regions = 0;
  tet_patch_workspace<Index> work;
};

/// State the regions of the opening `faces` names, read through `link`, the
/// manifold edge link of its triangles in the order `faces` blocks them —
/// the phase's own id space, so the partition is of the facets it is due.
/// Parallel at each carrier: the triangles, the facets, the sites and the
/// rows. O(triangles + facets + sites + rows) per opening, and nothing when
/// the surface is one region.
template <typename Owner, typename Faces, typename Link>
auto state_tet_patch_tables(
    const Owner &owner, const Faces &faces, const Link &link,
    tet_patch_tables<typename Owner::index_type> &tables,
    std::size_t target = tet_execution_tuning::region_triangles) -> void {
  using Index = typename Owner::index_type;
  const std::size_t n_facets = faces.size();
  auto &first = tables.facet_first;
  first.allocate(n_facets + 1);
  first[0] = 0;
  for (std::size_t f = 0; f < n_facets; ++f)
    first[f + 1] = first[f] + Index(faces[f].size());
  const std::size_t n_triangles = std::size_t(first[n_facets]);
  tf::buffer<Index> facet_of;
  facet_of.allocate(n_triangles);
  tf::parallel_for_each(
      tf::make_sequence_range(n_facets),
      [&facet_of, &first](std::size_t f) {
        for (Index t = first[f]; t < first[f + 1]; ++t)
          facet_of[std::size_t(t)] = Index(f);
      },
      tf::checked);
  const auto corners_of = [&faces, &first, &facet_of](Index t) {
    const std::size_t f = std::size_t(facet_of[std::size_t(t)]);
    return faces[f][std::size_t(t) - std::size_t(first[f])];
  };
  const auto facets = tf::make_offset_block_range(
      first, tf::make_sequence_range(Index(n_triangles)));
  const auto triangles = tf::make_mapped_range(
      tf::make_sequence_range(Index(n_triangles)), corners_of);
  tables.regions =
      partition_tet_surface(facets, triangles, tf::make_applier(link),
                            tables.triangle, tables.work, target);
  make_tet_facet_patches(facets, tables.triangle, tables.facet);
  make_tet_first_facet_patches(
      [&facets, &corners_of](std::size_t f, const auto &emit) {
        for (auto t : facets[f])
          for (auto corner : corners_of(t))
            emit(Index(corner));
      },
      tables.facet, tet_site_count(owner), tables.site, tables.work.claims);
}

} // namespace tf::topology::cdt::dt3

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
#include "../../../../../core/algorithm/parallel_fill.hpp"
#include "../../../../../core/algorithm/parallel_for_each.hpp"
#include "../../../../../core/algorithm/sequenced_generate.hpp"
#include "../../../../../core/buffer.hpp"
#include "../../../../../core/checked.hpp"
#include "../../tet_execution_tuning.hpp"
#include "./claim_tet_patch_ranges.hpp"
#include "./join_tet_patch_pockets.hpp"
#include "./tet_patch_workspace.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The triangles the ranges of the partition are drawn from, per patch.
inline constexpr std::size_t tet_patch_seed_samples = 256;

/// Partition the triangles `facets` names, per facet a range of ids into
/// `triangles`, each three corner slots, into patches of about `target`
/// triangles: a surface of fewer than twice `target` triangles is one patch,
/// a larger one `ceil(n / target)`. The build's Morton order of sites, read
/// as `{smallest corner slot, triangle}`, is cut into equal ranges at the
/// quantiles of a sample taken at a fixed stride of the facet order; patch
/// `p` claims the piece of range `p` its seed, the sample's median there,
/// reaches through `applier(t, f)` (@ref claim_tet_patch_ranges), and every
/// other triangle joins the patch it is fewest hops from
/// (@ref join_tet_patch_pockets). Every patch is connected and its size is
/// the target's only approximately; the labels are the same at any worker
/// count. `patch_of_triangle` holds per triangle its patch, `-1` for one no
/// facet names. Returns the patches, zero for an empty surface.
template <typename Facets, typename Triangles, typename Applier, typename Index>
auto partition_tet_surface(
    const Facets &facets, const Triangles &triangles, const Applier &applier,
    tf::buffer<Index> &patch_of_triangle, tet_patch_workspace<Index> &work,
    std::size_t target = tet_execution_tuning::region_triangles) -> Index {
  patch_of_triangle.allocate(triangles.size());
  tf::parallel_fill(patch_of_triangle, Index(-1));
  std::size_t n = 0;
  for (const auto &facet : facets)
    n += facet.size();
  if (n < 2 * target) {
    tf::parallel_for_each(
        facets,
        [&patch_of_triangle](const auto &facet) {
          for (auto t : facet)
            patch_of_triangle[std::size_t(t)] = Index(0);
        },
        tf::checked);
    return Index(n != 0);
  }
  auto &order = work.order;
  order.clear();
  tf::sequenced_generate(
      facets, order,
      [](const auto &facet, auto &out) {
        for (auto t : facet)
          out.push_back(Index(t));
      },
      tf::checked);
  const auto key_of = [&triangles](Index t) {
    const auto &corners = triangles[std::size_t(t)];
    return std::array<Index, 2>{
        Index(std::min({corners[0], corners[1], corners[2]})), t};
  };
  const std::size_t n_patches = (n + target - 1) / target;
  const std::size_t samples = std::min(n, n_patches * tet_patch_seed_samples);
  const std::size_t stride = n / samples;
  auto &keys = work.keys;
  keys.allocate(samples);
  for (std::size_t i = 0; i < samples; ++i)
    keys[i] = key_of(order[i * stride]);
  std::sort(keys.begin(), keys.end());
  auto &bounds = work.bounds;
  auto &seeds = work.seeds;
  bounds.allocate(n_patches - 1);
  seeds.allocate(n_patches);
  for (std::size_t p = 0; p < n_patches; ++p) {
    if (p)
      bounds[p - 1] = keys[p * samples / n_patches];
    seeds[p] = keys[(2 * p + 1) * samples / (2 * n_patches)][1];
  }
  auto &range = work.range;
  range.allocate(triangles.size());
  tf::parallel_for_each(
      order,
      [&range, &bounds, &key_of, &patch_of_triangle](Index t) {
        range[std::size_t(t)] =
            Index(std::upper_bound(bounds.begin(), bounds.end(), key_of(t)) -
                  bounds.begin());
        patch_of_triangle[std::size_t(t)] = Index(-2);
      },
      tf::checked);
  const auto range_of = [&range](Index t) { return range[std::size_t(t)]; };
  claim_tet_patch_ranges(seeds, range_of, applier, patch_of_triangle);
  join_tet_patch_pockets(order, range_of, applier, patch_of_triangle,
                         work.border, work.queue, work.hops);
  return Index(n_patches);
}

} // namespace tf::topology::cdt::dt3

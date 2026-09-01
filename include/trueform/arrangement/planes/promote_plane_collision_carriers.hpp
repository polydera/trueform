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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./plane_definition_source.hpp"
#include "./plane_tier_definitions.hpp"
#include "./state_plane_group_carriers.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::arrangement {

/// CORE. Whether one endpoint names a PROBED identity: a created identity and
/// nothing else, found in the ascending probe by one binary search.
template <typename Index>
auto probes_plane_identity(const tf::buffer<Index> &standing_probe,
                           std::int16_t tag, Index id) -> bool {
  if (tag != std::int16_t(-1))
    return false;
  const auto at =
      std::lower_bound(standing_probe.begin(), standing_probe.end(), id);
  return at != standing_probe.end() && *at == id;
}

/// Take every group a standing collision can reach, with its carriers.
///
/// THE OWNERSHIP LAW: a group whose statement CHANGES goes local and takes
/// every carrier of it in the same wave. A key this wave states can equal a
/// standing key only through a PRE-EXISTING identity — a cut or a merge target
/// — so the rows naming those identities name every group the canonicalize can
/// possibly fuse, and a fusion is a change. Each such group joins the wave's
/// TAKEN set and its whole instance span joins the port frontier; by the time
/// keys close, every collision partner lives in this arrangement's tier and no
/// carrier of it still reads the world.
///
/// The sweep is one flat pass at PLANE grain over the block each plane's
/// ticket names, resolved through the flat row space — a row still the world's
/// names a group the port must take, and one already local names a group an
/// earlier wave took, whose carriers are already here.
///
/// A group that does not finally fuse costs one verbatim span copy made early
/// — a valid local group either way. Taking is not retriangulation: a carrier
/// keeps its product, a refusal included, because a verbatim span changes no
/// row it would read.
template <typename Index, typename Int, typename PlaneOfFace>
auto promote_plane_collision_carriers(
    const tf::intersect::graph::plane_tables<Index, Int> &world_tables,
    const tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    const tf::buffer<Index> &plane_ticket,
    const tf::buffer<Index> &standing_probe, const PlaneOfFace &plane_of_face,
    tf::buffer<Index> &frontier, tf::buffer<Index> &taken) -> void {
  if (standing_probe.size() == 0)
    return;
  const auto n_planes = Index(plane_ticket_space(world_tables, plane_ticket));
  tf::buffer<Index> roots;
  tf::generic_generate(
      tf::make_sequence_range(n_planes), roots,
      [&](Index plane, tf::buffer<Index> &out) {
        const auto source = find_plane_definition_source(
            world_tables, local_tables, plane_ticket, plane);
        const auto tier = make_plane_tier_definitions(
            world_tables, local_tables, !source.immutable);
        for (const auto row : source.tables.plane_edges(source.block)) {
          if (!tier.immutable(row))
            continue;
          const auto &def = tier[std::size_t(row)];
          if (probes_plane_identity(standing_probe, def.point_tag_0,
                                    def.point_0) ||
              probes_plane_identity(standing_probe, def.point_tag_1,
                                    def.point_1))
            out.push_back(def.id);
        }
      });
  if (roots.size() == 0)
    return;
  tbb::parallel_sort(roots.begin(), roots.end());
  roots.erase_till_end(std::unique(roots.begin(), roots.end()));

  tf::buffer<Index> carriers;
  state_plane_group_carriers(world_tables, tf::make_range(roots), plane_of_face,
                             carriers);
  tf::core::append(roots, taken);
  tf::core::append(frontier, carriers);
  tbb::parallel_sort(carriers.begin(), carriers.end());
  carriers.erase_till_end(std::unique(carriers.begin(), carriers.end()));
  frontier = std::move(carriers);
}

} // namespace tf::arrangement

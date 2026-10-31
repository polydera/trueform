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
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/intersect_config.hpp"
#include "./make_carve_package.hpp"
#include "./reseat_census.hpp"
#include "./reseat_pieces.hpp"
#include "./state_reseat_targets.hpp"
#include <array>

namespace tf::arrangement::reseat {

/// Carve every carrier that does not hold the position the gate fused its
/// identity onto into the built `world` (@ref make_carve_package), and state
/// the fixed point into `graph` (`state_reseat`). A carve that cannot be
/// stated is refused in the census and the world stands as built.
template <typename Index, typename Int, typename World, typename ApplyToForm,
          typename Reader, typename Converter, typename Graph>
auto carve_world(const World &world, const ApplyToForm &apply_to_form,
                 const Reader &reader, const Converter &converter, Index n_tags,
                 tf::intersect_config config, Graph &graph) -> void {
  const auto &identities = graph.identities();
  if (identities.local_merges().size() == 0)
    return;
  const auto quantum = graph.quantum();
  const auto created = tf::make_mapped_range(
      tf::make_sequence_range(world.n_created_points(identities)),
      [&](auto id) { return world.point_of(identities, id, reader); });
  const standing_pieces<Index> standing;
  const auto targets = state_reseat_targets<Index, Int>(
      world, tf::make_range(identities.local_merges()), created, standing,
      apply_to_form, reader, quantum);
  if (targets.size() == 0)
    return;
  carve_reach_census reach;
  const auto package = make_carve_package<Index, Int>(
      world, graph.identities(), standing, created, targets, apply_to_form,
      reader, converter, n_tags, quantum, config, [](auto &) {},
      [](auto, auto) { return false; }, reach);
  if (!package.ok) {
    graph.policy().refuse_carve(package.refusal, reach);
    return;
  }
  tf::buffer<Index> row_images;
  tf::buffer<Index> canon_images;
  tf::buffer<Index> touched_planes;
  tf::buffer<std::array<Index, 3>> moved_rows;
  graph.state_reseat(package, row_images, canon_images, touched_planes,
                     moved_rows);
  graph.policy().count_carve(reach);
}

} // namespace tf::arrangement::reseat
